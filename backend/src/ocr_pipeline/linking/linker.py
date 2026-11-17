"""Verify each LLM source against the real page text and locate it on the page."""

from __future__ import annotations

from ocr_pipeline.config import LinkingSettings
from ocr_pipeline.linking.paths import covers, leaf_paths, resolve
from ocr_pipeline.linking.text import PageIndex, find_quote, line_boxes, matched_words, tokens
from ocr_pipeline.linking.values import value_in_text
from ocr_pipeline.models.document import ExtractedDocument
from ocr_pipeline.models.extraction import (
    ExtractionResult,
    SourceRef,
    SourceStatus,
    SourceVerification,
)


def _locate(
    quote: list[str], page_no: int, indexes: dict[int, PageIndex], settings: LinkingSettings
) -> tuple[int, PageIndex, object] | None:
    """Best match on the cited page; otherwise (optionally) the best on any other page."""
    cited = indexes.get(page_no)
    if cited is not None:
        m = find_quote(quote, cited, settings.fuzzy_threshold)
        if m is not None:
            return page_no, cited, m
    if not settings.search_other_pages:
        return None
    best = None
    for other_no, index in indexes.items():
        if other_no == page_no:
            continue
        m = find_quote(quote, index, settings.fuzzy_threshold)
        if m is not None and (best is None or m.score > best[2].score):
            best = (other_no, index, m)
    return best


def verify_source(
    source: SourceRef,
    data: dict,
    indexes: dict[int, PageIndex],
    settings: LinkingSettings,
    date_order: str = "auto",
) -> SourceVerification:
    field_found, value = resolve(data, source.field)
    found = _locate(tokens(source.quote), source.page, indexes, settings)

    if found is None:
        status = SourceStatus.UNKNOWN_FIELD if not field_found else SourceStatus.NOT_FOUND
        return SourceVerification(status=status, score=0.0, value=value)

    page_no, index, match = found
    words = matched_words(match, index)
    text = " ".join(w.text for w in words)
    if not field_found:
        status = SourceStatus.UNKNOWN_FIELD
    elif page_no != source.page:
        status = SourceStatus.WRONG_PAGE
    elif match.score == 1.0:
        status = SourceStatus.VERIFIED
    else:
        status = SourceStatus.FUZZY
    return SourceVerification(
        status=status,
        score=round(match.score, 4),
        page=page_no,
        matched_text=text,
        boxes=line_boxes(words),
        value=value,
        # Checked against the real page text, not the model's quote.
        value_found=value_in_text(value, text, date_order) if field_found else None,
    )


def link_sources(
    document: ExtractedDocument,
    result: ExtractionResult,
    settings: LinkingSettings,
    date_order: str = "auto",
) -> ExtractionResult:
    """Return a copy of ``result`` with every source verified and unsourced fields listed.

    ``date_order`` ("DMY"/"MDY") makes date checks strict, so a swapped day and month
    (12/11/2026 read as 11 December in a DD/MM document) is flagged.
    """
    indexes = {p.page_number: PageIndex.build(p.words) for p in document.pages if p.words}
    sources = [
        s.model_copy(
            update={"verification": verify_source(s, result.data, indexes, settings, date_order)}
        )
        for s in result.sources
    ]
    cited = [s.field for s in sources if s.verification.status != SourceStatus.UNKNOWN_FIELD]
    unsourced = [leaf for leaf in leaf_paths(result.data) if not any(covers(c, leaf) for c in cited)]
    return result.model_copy(update={"sources": sources, "unsourced_fields": unsourced})
