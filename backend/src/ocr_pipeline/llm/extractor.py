"""LLM extraction: chunk the document if needed, call the provider, validate, merge."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from ocr_pipeline.llm.prompt import SYSTEM_PROMPT, retry_prompt, user_prompt
from ocr_pipeline.llm.providers.base import LLMError, LLMProvider
from ocr_pipeline.llm.schema import FREE_FORM_SCHEMA, TargetSchema, envelope, to_strict_schema
from ocr_pipeline.models.document import ExtractedDocument, PageContent
from ocr_pipeline.models.extraction import ExtractionResult, FieldConflict, SourceRef

CHARS_PER_TOKEN = 3.5  # conservative estimate; real tokenizers vary by provider
PROMPT_OVERHEAD_TOKENS = 1_500  # system prompt + instructions + schema


class ExtractionError(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def input_overhead(wire_schema: dict, instructions: str | None) -> int:
    """Estimated tokens used by everything except the page text."""
    return PROMPT_OVERHEAD_TOKENS + estimate_tokens(json.dumps(wire_schema) + (instructions or ""))


def chunk_pages(pages: list[PageContent], budget: int) -> list[list[PageContent]]:
    """Group consecutive pages so each group fits ``budget`` estimated tokens.

    Pages are never split or truncated; a single page over budget is an error.
    """
    chunks: list[list[PageContent]] = []
    current: list[PageContent] = []
    used = 0
    for page in pages:
        if not page.markdown.strip():
            continue
        cost = estimate_tokens(page.markdown) + 20  # page tags
        if cost > budget:
            raise ExtractionError(
                f"page {page.page_number} alone needs ~{cost} tokens, over the input budget of "
                f"{budget}; raise max_input_tokens for this provider"
            )
        if current and used + cost > budget:
            chunks.append(current)
            current, used = [], 0
        current.append(page)
        used += cost
    if current:
        chunks.append(current)
    return chunks


def _merge(a: Any, b: Any, path: str, pages_a: list[int], pages_b: list[int], conflicts: dict) -> Any:
    if a is None or a == [] or a == {}:
        return b
    if b is None or b == [] or b == {}:
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        keys = list(dict.fromkeys([*a, *b]))
        return {
            k: _merge(a.get(k), b.get(k), f"{path}.{k}" if path else k, pages_a, pages_b, conflicts)
            for k in keys
        }
    if isinstance(a, list) and isinstance(b, list):
        return a + b  # e.g. line items continuing on the next page
    if a == b:
        return a
    c = conflicts.setdefault(path, FieldConflict(field=path, values=[a], pages=[pages_a]))
    c.values.append(b)
    c.pages.append(pages_b)
    return a  # keep the first value; the conflict is reported


def _dedupe_free_form(fields: list[dict], conflicts: dict, pages_of: list[list[int]]) -> list[dict]:
    """Free-form mode: same name from several chunks -> keep first, report differences."""
    seen: dict[str, int] = {}
    out: list[dict] = []
    for f, pages in zip(fields, pages_of, strict=True):
        name = f["name"]
        if name not in seen:
            seen[name] = len(out)
            out.append(f)
            continue
        first = out[seen[name]]
        if first["value"] is None:
            first["value"] = f["value"]
        elif f["value"] is not None and f["value"] != first["value"]:
            c = conflicts.setdefault(name, FieldConflict(field=name, values=[first["value"]], pages=[[]]))
            c.values.append(f["value"])
            c.pages.append(pages)
    return out


def extract(
    document: ExtractedDocument,
    provider: LLMProvider,
    schema: type[BaseModel] | dict | None = None,
    instructions: str | None = None,
    max_retries: int = 1,
    date_order: str = "auto",
) -> ExtractionResult:
    """Extract data from ``document`` per ``schema`` and/or free-text ``instructions``.

    ``date_order`` ("DMY"/"MDY"/"YMD") tells the model how numeric dates are written.
    """
    if schema is None and not (instructions and instructions.strip()):
        raise ValueError("give a schema, instructions, or both")

    target = TargetSchema(schema) if schema is not None else None
    free_form = target is None
    data_schema = target.strict if target else to_strict_schema(FREE_FORM_SCHEMA)
    wire_schema = envelope(data_schema)

    budget = provider.max_input_tokens - input_overhead(wire_schema, instructions)
    if budget <= 0:
        raise ExtractionError("schema and instructions alone exceed the input budget")
    chunks = chunk_pages(document.pages, budget)
    if not chunks:
        raise ExtractionError("the document has no text to extract from")

    warnings: list[str] = []
    conflicts: dict[str, FieldConflict] = {}
    sources: list[SourceRef] = []
    merged: Any = None
    merged_pages: list[int] = []
    free_fields: list[dict] = []
    free_pages: list[list[int]] = []
    tokens_in = tokens_out = 0
    model = provider.model

    for i, chunk in enumerate(chunks):
        page_numbers = [p.page_number for p in chunk]
        prompt = user_prompt(chunk, instructions, i, len(chunks), free_form, date_order)
        attempt_prompt = prompt
        for attempt in range(max_retries + 1):
            try:
                response = provider.complete_json(SYSTEM_PROMPT, attempt_prompt, wire_schema)
            except LLMError as e:
                raise ExtractionError(f"{provider.name}: {e}") from e
            tokens_in += response.input_tokens
            tokens_out += response.output_tokens
            model = response.model
            try:
                payload = json.loads(response.text)
                data = payload["data"]
                chunk_sources = [SourceRef.model_validate(s) for s in payload.get("sources", [])]
                if target is not None:
                    data = target.validate(data)
                break
            except (ValueError, KeyError, TypeError) as e:  # JSONDecodeError is a ValueError
                if attempt == max_retries:
                    raise ExtractionError(
                        f"pages {page_numbers[0]}-{page_numbers[-1]}: output failed validation "
                        f"after {max_retries + 1} attempt(s): {e}"
                    ) from e
                attempt_prompt = prompt + retry_prompt(response.text, str(e))

        for s in chunk_sources:
            if s.page not in page_numbers:
                warnings.append(f"source for {s.field!r} cites page {s.page}, not in pages {page_numbers}")
        sources.extend(chunk_sources)

        if free_form:
            free_fields.extend(data.get("fields", []))
            free_pages.extend([page_numbers] * len(data.get("fields", [])))
        else:
            merged = _merge(merged, data, "", merged_pages, page_numbers, conflicts)
            merged_pages = merged_pages + page_numbers

    if free_form:
        merged = {"fields": _dedupe_free_form(free_fields, conflicts, free_pages)}
    elif len(chunks) > 1:
        try:
            merged = target.validate(merged)
        except ValueError as e:
            warnings.append(f"merged result does not fully match the schema: {e}")

    return ExtractionResult(
        data=merged,
        sources=sources,
        conflicts=list(conflicts.values()),
        warnings=warnings,
        provider=provider.name,
        model=model,
        chunks=len(chunks),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
    )
