"""Per-page classification: digital / scanned / empty, and whether OCR is needed."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from ocr_pipeline.classify.quality import collect_signals, quality_score
from ocr_pipeline.config import ClassifierSettings
from ocr_pipeline.models.document import (
    ClassificationReason as Reason,
    PageClassification,
    PageSignals,
    PageType,
)


def decide(
    page_number: int, signals: PageSignals, settings: ClassifierSettings
) -> PageClassification:
    """Pure decision logic: signals + settings -> classification."""

    def result(
        page_type: PageType,
        needs_ocr: bool,
        reason: Reason,
        score: float | None = None,
        components: dict[str, float] | None = None,
    ) -> PageClassification:
        return PageClassification(
            page_number=page_number,
            page_type=page_type,
            needs_ocr=needs_ocr,
            reason=reason,
            quality_score=score,
            threshold=settings.quality_threshold,
            score_components=components or {},
            signals=signals,
        )

    # 1. No text at all: a scan if anything is drawn or printed, else blank.
    if signals.char_count == 0:
        if signals.image_coverage > 0 or signals.curve_vector_count > 0:
            return result(PageType.SCANNED, True, Reason.NO_TEXT_LAYER)
        if (signals.ink_ratio or 0) > settings.empty_max_ink:
            # Visible, but not extractable: OCR the rendered page.
            return result(PageType.SCANNED, True, Reason.UNEXTRACTABLE_CONTENT)
        return result(PageType.EMPTY, False, Reason.EMPTY_PAGE)

    # 2. A big image with only a little real text (e.g. a stamp or Bates number).
    if (
        not signals.has_ocr_layer
        and signals.image_coverage >= settings.image_dominant_coverage
        and signals.visible_char_count < settings.image_dominant_max_chars
    ):
        return result(PageType.SCANNED, True, Reason.IMAGE_DOMINANT_SPARSE_TEXT)

    # 3. Score the text layer; too many broken characters fails regardless of score.
    score, components = quality_score(signals, settings)
    low = (score is not None and score < settings.quality_threshold) or (
        signals.bad_char_ratio > settings.max_bad_char_ratio
    )

    if signals.has_ocr_layer:
        reason = Reason.OCR_LAYER_LOW_QUALITY if low else Reason.OCR_LAYER_OK
        return result(PageType.SCANNED, low, reason, score, components)

    reason = Reason.TEXT_LAYER_LOW_QUALITY if low else Reason.TEXT_LAYER_OK
    return result(PageType.DIGITAL, low, reason, score, components)


def classify_page(page: pymupdf.Page, settings: ClassifierSettings) -> PageClassification:
    return decide(page.number + 1, collect_signals(page, settings), settings)


def classify_document(
    source: str | Path | pymupdf.Document, settings: ClassifierSettings
) -> list[PageClassification]:
    """Classify every page. Accepts a path or an already-open document."""
    if isinstance(source, pymupdf.Document):
        return [classify_page(page, settings) for page in source]
    with pymupdf.open(source) as doc:
        return [classify_page(page, settings) for page in doc]
