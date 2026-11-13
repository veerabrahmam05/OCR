"""Extraction logic with a fake provider: prompting, chunking, retries, merging."""

from __future__ import annotations

import pytest
from llm_fakes import FakeProvider, make_document

from ocr_pipeline.llm.extractor import (
    ExtractionError,
    estimate_tokens,
    extract,
    input_overhead,
)
from ocr_pipeline.llm.providers.base import LLMRefusal
from ocr_pipeline.llm.schema import FREE_FORM_SCHEMA, TargetSchema, envelope, to_strict_schema
from ocr_pipeline.schemas import Invoice

PAGE1 = "# Invoice INV-2026-0042\n\nDate: 2026-11-12\n\n|Item|Qty|Amount|\n|---|---|---|\n|Steel bolts|200|100.00|"
PAGE2 = "|Copper wire|15|180.00|\n\nTotal due: 280.00 USD"


def invoice_reply(**data):
    base = {"invoice_number": "INV-2026-0042", "invoice_date": "2026-11-12", "line_items": []}
    return {
        "data": base | data,
        "sources": [{"field": "invoice_number", "page": 1, "quote": "Invoice INV-2026-0042"}],
    }


def one_page_per_chunk(doc, schema=Invoice, instructions=None) -> int:
    """A max_input_tokens value that fits exactly one of the test pages per chunk."""
    wire = envelope(TargetSchema(schema).strict if schema else to_strict_schema(FREE_FORM_SCHEMA))
    biggest = max(estimate_tokens(p.markdown) + 20 for p in doc.pages)
    return input_overhead(wire, instructions) + biggest + 5


# --- single call ------------------------------------------------------------------


def test_single_call_with_schema():
    doc = make_document([PAGE1, PAGE2])
    provider = FakeProvider([invoice_reply(total=280.0)])
    result = extract(doc, provider, Invoice, instructions="Amounts are in USD.")

    assert len(provider.calls) == 1
    user = provider.calls[0]["user"]
    assert '<page number="1">' in user and '<page number="2">' in user
    assert "<instructions>\nAmounts are in USD.\n</instructions>" in user
    assert "part 1 of" not in user
    assert provider.calls[0]["schema"]["required"] == ["data", "sources"]

    assert result.data["invoice_number"] == "INV-2026-0042"
    assert result.data["total"] == 280.0
    assert result.sources[0].page == 1
    assert (result.provider, result.model, result.chunks) == ("fake", "fake-model-v1", 1)
    assert (result.input_tokens, result.output_tokens) == (100, 20)
    assert result.conflicts == [] and result.warnings == []


def test_instructions_only_uses_free_form_fields():
    doc = make_document([PAGE1])
    reply = {
        "data": {"fields": [{"name": "invoice_number", "value": "INV-2026-0042"}]},
        "sources": [{"field": "invoice_number", "page": 1, "quote": "INV-2026-0042"}],
    }
    provider = FakeProvider([reply])
    result = extract(doc, provider, instructions="Get the invoice number.")
    assert result.data == {"fields": [{"name": "invoice_number", "value": "INV-2026-0042"}]}
    assert "data.fields" in provider.calls[0]["user"]


def test_schema_or_instructions_required():
    with pytest.raises(ValueError):
        extract(make_document([PAGE1]), FakeProvider([]))


def test_empty_document_rejected():
    with pytest.raises(ExtractionError, match="no text"):
        extract(make_document(["", "  "]), FakeProvider([]), Invoice)


def test_empty_pages_are_left_out_of_the_prompt():
    provider = FakeProvider([invoice_reply()])
    extract(make_document([PAGE1, "", PAGE2]), provider, Invoice)
    user = provider.calls[0]["user"]
    assert '<page number="2">' not in user and '<page number="3">' in user


# --- retries and failures -----------------------------------------------------------


def test_invalid_output_is_retried_with_the_error():
    provider = FakeProvider([invoice_reply(invoice_date="not a date"), invoice_reply()])
    result = extract(make_document([PAGE1]), provider, Invoice)
    assert len(provider.calls) == 2
    retry = provider.calls[1]["user"]
    assert "<validation_error>" in retry and "invoice_date" in retry
    assert result.data["invoice_date"] == "2026-11-12"


def test_malformed_json_is_retried():
    provider = FakeProvider(['{"data": ', invoice_reply()])
    assert extract(make_document([PAGE1]), provider, Invoice).data["invoice_number"]


def test_gives_up_after_max_retries():
    provider = FakeProvider([invoice_reply(invoice_date="bad")] * 3)
    with pytest.raises(ExtractionError, match="after 3 attempt"):
        extract(make_document([PAGE1]), provider, Invoice, max_retries=2)
    assert len(provider.calls) == 3


def test_provider_refusal_is_reported():
    provider = FakeProvider([LLMRefusal("declined (category: cyber)")])
    with pytest.raises(ExtractionError, match="fake: declined"):
        extract(make_document([PAGE1]), provider, Invoice)


def test_source_citing_page_outside_chunk_gives_warning():
    reply = invoice_reply()
    reply["sources"] = [{"field": "invoice_number", "page": 7, "quote": "INV"}]
    result = extract(make_document([PAGE1]), FakeProvider([reply]), Invoice)
    assert "page 7" in result.warnings[0]


# --- chunking & merging --------------------------------------------------------------


def test_long_document_is_chunked_by_page():
    doc = make_document([PAGE1, PAGE2])
    provider = FakeProvider(
        [
            invoice_reply(line_items=[{"description": "Steel bolts", "quantity": 200, "amount": 100.0}]),
            invoice_reply(
                line_items=[{"description": "Copper wire", "quantity": 15, "amount": 180.0}], total=280.0
            ),
        ],
        max_input_tokens=one_page_per_chunk(doc),
    )
    result = extract(doc, provider, Invoice)

    assert result.chunks == 2 and len(provider.calls) == 2
    assert "part 1 of 2" in provider.calls[0]["user"] and "(pages 1-1)" in provider.calls[0]["user"]
    assert "part 2 of 2" in provider.calls[1]["user"]
    assert [i["description"] for i in result.data["line_items"]] == ["Steel bolts", "Copper wire"]
    assert result.data["total"] == 280.0  # null in chunk 1, filled by chunk 2
    assert result.data["invoice_number"] == "INV-2026-0042"  # same in both: no conflict
    assert result.conflicts == []
    assert len(result.sources) == 2


def test_conflicting_values_across_chunks_are_reported():
    doc = make_document([PAGE1, PAGE2])
    provider = FakeProvider(
        [invoice_reply(total=100.0), invoice_reply(total=280.0)],
        max_input_tokens=one_page_per_chunk(doc),
    )
    result = extract(doc, provider, Invoice)
    assert result.data["total"] == 100.0  # first value kept
    (conflict,) = result.conflicts
    assert conflict.field == "total"
    assert conflict.values == [100.0, 280.0]
    assert conflict.pages == [[1], [2]]


def test_free_form_fields_deduplicated_across_chunks():
    doc = make_document([PAGE1, PAGE2])

    def reply(fields):
        return {"data": {"fields": [{"name": n, "value": v} for n, v in fields]}, "sources": []}

    provider = FakeProvider(
        [
            reply([("invoice_number", "INV-2026-0042"), ("total", None)]),
            reply([("invoice_number", "INV-2026-0042"), ("total", "280.00"), ("vendor", "Acme")]),
        ],
        max_input_tokens=one_page_per_chunk(doc, schema=None, instructions="all"),
    )
    result = extract(doc, provider, instructions="all")
    assert result.data["fields"] == [
        {"name": "invoice_number", "value": "INV-2026-0042"},
        {"name": "total", "value": "280.00"},
        {"name": "vendor", "value": "Acme"},
    ]
    assert result.conflicts == []


def test_page_bigger_than_budget_is_an_error_not_truncated():
    doc = make_document([PAGE1 * 50])
    provider = FakeProvider([], max_input_tokens=one_page_per_chunk(make_document([PAGE1])))
    with pytest.raises(ExtractionError, match="page 1 alone"):
        extract(doc, provider, Invoice)
