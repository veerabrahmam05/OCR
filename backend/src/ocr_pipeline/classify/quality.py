"""Text-layer signals and the quality score for a single PDF page."""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from functools import lru_cache

import numpy as np
import pymupdf
from pymupdf import mupdf
from spellchecker import SpellChecker

from ocr_pipeline.config import ClassifierSettings
from ocr_pipeline.models.document import PageSignals

log = logging.getLogger(__name__)

# Same flags pymupdf4llm uses for its OCR decision, so its feature extractor
# can reuse the blocks we already extracted.
DICT_FLAGS = (
    pymupdf.TEXT_ACCURATE_BBOXES | pymupdf.TEXT_PRESERVE_IMAGES | pymupdf.TEXT_COLLECT_VECTORS
)
BLOCK_TEXT = mupdf.FZ_STEXT_BLOCK_TEXT
BLOCK_IMAGE = mupdf.FZ_STEXT_BLOCK_IMAGE
BLOCK_VECTOR = mupdf.FZ_STEXT_BLOCK_VECTOR
TEXT_VISIBLE_FLAGS = mupdf.FZ_STEXT_STROKED | mupdf.FZ_STEXT_FILLED
TESSERACT_FONT = "GlyphLessFont"

_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_BAD_CATEGORIES = {"Cc", "Co", "Cs", "Cn"}  # control, private-use, surrogate, unassigned
_COVERAGE_GRID = 100  # image coverage is measured on a 100x100 grid


def _is_ocr_span(span: dict) -> bool:
    """Invisible text: Tesseract's font, fully transparent, or neither filled nor stroked."""
    return (
        span.get("font") == TESSERACT_FONT
        or span.get("alpha", 255) == 0
        or span.get("char_flags", mupdf.FZ_STEXT_FILLED) & TEXT_VISIBLE_FLAGS == 0
    )


def _is_bad_char(c: str) -> bool:
    return c == "�" or unicodedata.category(c) in _BAD_CATEGORIES


def _union_coverage(rects: Iterable[Sequence[float]], page_rect: pymupdf.Rect) -> float:
    """Fraction of the page covered by the union of ``rects`` (overlaps counted once)."""
    if page_rect.is_empty:
        return 0.0
    n = _COVERAGE_GRID
    grid = np.zeros((n, n), dtype=bool)
    sx, sy = n / page_rect.width, n / page_rect.height
    for r in rects:
        r = pymupdf.Rect(r) & page_rect
        if r.is_empty:
            continue
        x0 = int((r.x0 - page_rect.x0) * sx)
        y0 = int((r.y0 - page_rect.y0) * sy)
        x1 = math.ceil((r.x1 - page_rect.x0) * sx)
        y1 = math.ceil((r.y1 - page_rect.y0) * sy)
        grid[y0:y1, x0:x1] = True
    return float(grid.mean())


@lru_cache(maxsize=1)
def _spellchecker() -> SpellChecker:
    return SpellChecker(language="en")


def dictionary_ratio(text: str, min_words: int) -> tuple[float | None, int]:
    """Share of alphabetic words that are English; None if too few words to judge."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) < min_words:
        return None, len(words)
    known = _spellchecker().known(set(words))
    return sum(w in known for w in words) / len(words), len(words)


def _ocr_model_probability(blocks: list[dict], page: pymupdf.Page) -> float | None:
    """Probability from pymupdf4llm's OCR-decision model that the page needs OCR.

    Uses pymupdf4llm internals (not public API), so any failure degrades to None
    and the component is simply left out of the score.
    """
    try:
        from pymupdf4llm.ocr.analyze_page import predict_ocr_probability
        from pymupdf4llm.ocr.compute_ocr_features import compute_features

        return predict_ocr_probability(compute_features(blocks, page.rect, page))
    except Exception:
        log.warning("pymupdf4llm OCR decision model unavailable; skipping", exc_info=True)
        return None


def collect_signals(page: pymupdf.Page, settings: ClassifierSettings) -> PageSignals:
    blocks = page.get_text("dict", flags=DICT_FLAGS)["blocks"]

    texts: list[str] = []
    char_count = ocr_char_count = bad_chars = curve_vectors = 0
    image_rects: list[Sequence[float]] = []

    for b in blocks:
        btype = b["type"]
        if btype == BLOCK_TEXT:
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    n = len(text)
                    char_count += n
                    if _is_ocr_span(span):
                        ocr_char_count += n
                    bad_chars += sum(_is_bad_char(c) for c in text)
                    texts.append(text)
        elif btype == BLOCK_IMAGE:
            image_rects.append(b["bbox"])
        elif btype == BLOCK_VECTOR and not b.get("isrect", True):
            curve_vectors += 1

    has_ocr_layer = char_count > 0 and ocr_char_count / char_count >= settings.ocr_layer_min_fraction
    dict_ratio, word_count = dictionary_ratio(" ".join(texts), settings.min_words_for_dictionary)

    # The model detects text hidden in images/drawings that the text layer misses.
    # Skip it for OCR-layer pages: their image always contains text by definition.
    model_prob = None
    if (
        settings.use_ocr_model
        and char_count
        and not has_ocr_layer
        and (image_rects or curve_vectors)
    ):
        model_prob = _ocr_model_probability(blocks, page)

    # Nothing extractable at all: look at the rendered page before calling it empty.
    # Content can live where extraction does not see it (annotation appearances,
    # patterns, shadings) while still being printed on the page.
    ink = None
    if char_count == 0 and not image_rects and not curve_vectors:
        ink = ink_ratio(page)

    return PageSignals(
        char_count=char_count,
        ocr_char_count=ocr_char_count,
        word_count=word_count,
        bad_char_ratio=bad_chars / char_count if char_count else 0.0,
        dictionary_ratio=dict_ratio,
        image_coverage=_union_coverage(image_rects, page.rect),
        curve_vector_count=curve_vectors,
        has_ocr_layer=has_ocr_layer,
        ocr_model_probability=model_prob,
        ink_ratio=ink,
    )


INK_DPI = 36  # low resolution is enough to tell blank from printed
INK_LEVEL = 200  # gray value below which a pixel counts as ink


def ink_ratio(page: pymupdf.Page) -> float:
    """Fraction of non-white pixels when the page (with annotations) is rendered."""
    pix = page.get_pixmap(dpi=INK_DPI, colorspace=pymupdf.csGRAY, annots=True)
    gray = np.frombuffer(pix.samples, dtype=np.uint8)
    return float((gray < INK_LEVEL).mean()) if gray.size else 0.0


def quality_score(
    signals: PageSignals, settings: ClassifierSettings
) -> tuple[float | None, dict[str, float]]:
    """Weighted average of the available components, each in [0, 1].

    Components that cannot be measured (None) are dropped and the remaining
    weights renormalised. Returns (None, {}) when nothing can be measured.
    """
    if signals.char_count == 0:
        return None, {}

    candidates: dict[str, float | None] = {
        "dictionary": signals.dictionary_ratio,
        "clean_chars": max(0.0, 1.0 - signals.bad_char_ratio * settings.bad_char_penalty),
        "image_text": (
            None if signals.ocr_model_probability is None else 1.0 - signals.ocr_model_probability
        ),
    }
    components = {k: v for k, v in candidates.items() if v is not None}
    weights = settings.weights.model_dump()
    total = sum(weights[k] for k in components)
    if total <= 0:
        return None, components
    score = sum(components[k] * weights[k] for k in components) / total
    return score, components
