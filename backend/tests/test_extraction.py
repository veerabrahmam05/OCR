"""Extraction tests: digital and OCR pages end up in the same Markdown shape.

These run real Tesseract and are skipped if its language data is not installed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from pdf_builders import (
    ENGLISH,
    add_digital,
    add_digital_invoice,
    add_scanned,
    draw_invoice,
    draw_text_as_rectangles,
    gibberish,
)

from ocr_pipeline.classify import classify_document
from ocr_pipeline.config import OcrSettings, Settings, load_settings
from ocr_pipeline.extract import TesseractEngine, create_engine, extract_pages
from ocr_pipeline.extract.preprocess import remove_ruling_lines, to_rgb
from ocr_pipeline.models import ExtractionMethod, PageType
from ocr_pipeline.pipeline import process_pdf

REPO_CONFIG = Path(__file__).parents[1] / "config" / "default.toml"


@pytest.fixture(scope="module")
def settings() -> Settings:
    s = load_settings(REPO_CONFIG)
    try:
        create_engine(s.ocr)
    except FileNotFoundError as e:
        pytest.skip(f"Tesseract not available: {e}")
    return s


def _norm(text: str) -> str:
    """Lowercase, drop Markdown emphasis, collapse whitespace."""
    return re.sub(r"\s+", " ", text.replace("*", "")).strip().lower()


# One mixed document processed once; the tests below inspect its pages.
PAGES = [
    "digital text",
    "digital ruled invoice",
    "scanned text",
    "scanned ruled invoice",
    "scanned borderless invoice",
    "garbage text layer",
    "empty",
]


@pytest.fixture(scope="module")
def mixed_pdf() -> bytes:
    with pymupdf.open() as d:
        add_digital(d)
        add_digital_invoice(d, ruled=True)
        add_scanned(d)
        add_scanned(d, draw=lambda p: draw_invoice(p, ruled=True))
        add_scanned(d, draw=lambda p: draw_invoice(p, ruled=False))
        add_digital(d, gibberish())
        d.new_page()
        return d.tobytes()


@pytest.fixture(scope="module")
def result(mixed_pdf, settings):
    return process_pdf(mixed_pdf, settings)


def page(result, label: str):
    return result.pages[PAGES.index(label)]


# --- routing ----------------------------------------------------------------------


def test_every_page_is_returned_in_order(result):
    assert result.source == "<bytes>"
    assert result.page_count == len(PAGES)
    assert [p.page_number for p in result.pages] == list(range(1, len(PAGES) + 1))


@pytest.mark.parametrize(
    "label, method",
    [
        ("digital text", ExtractionMethod.TEXT_LAYER),
        ("digital ruled invoice", ExtractionMethod.TEXT_LAYER),
        ("scanned text", ExtractionMethod.OCR),
        ("scanned ruled invoice", ExtractionMethod.OCR),
        ("scanned borderless invoice", ExtractionMethod.OCR),
        ("garbage text layer", ExtractionMethod.OCR),
        ("empty", ExtractionMethod.NONE),
    ],
)
def test_extraction_method_per_page(result, label, method):
    p = page(result, label)
    assert p.method == method
    assert p.ocr_engine == ("tesseract" if method == ExtractionMethod.OCR else None)


# --- content ----------------------------------------------------------------------


def test_digital_and_scanned_text_give_the_same_content(result):
    digital = _norm(page(result, "digital text").markdown)
    scanned = _norm(page(result, "scanned text").markdown)
    for line in ENGLISH:
        assert _norm(line) in digital
    # OCR may slip on a character; require nearly all words to match.
    words = _norm(" ".join(ENGLISH)).split()
    found = sum(w in scanned for w in words)
    assert found / len(words) > 0.95


@pytest.mark.parametrize(
    "label", ["digital ruled invoice", "scanned ruled invoice", "scanned borderless invoice"]
)
def test_tables_keep_their_rows_and_columns(result, label):
    md = page(result, label).markdown.replace("*", "")
    assert "table" in [b.kind for b in page(result, label).blocks]
    assert "|Item|Qty|Unit price|Amount|" in md
    assert "|Steel bolts|200|0.50|100.00|" in md
    assert "|Copper wire|15|12.00|180.00|" in md
    assert re.search(r"\|Safety ?gloves\|40\|3\.25\|130\.00\|", md)  # OCR may drop the space


def test_empty_page_has_no_content(result):
    p = page(result, "empty")
    assert p.markdown.strip() == ""
    assert p.blocks == []
    assert p.classification.page_type == PageType.EMPTY


def test_document_markdown_joins_pages(result):
    md = result.markdown
    assert "Steel bolts" in md and "arbitration" in md


# --- blocks (basis for source linking) ----------------------------------------------


def test_blocks_point_into_page_markdown(result):
    for p in result.pages:
        for b in p.blocks:
            assert b.text == p.markdown[b.start : b.end]
            x0, y0, x1, y1 = b.bbox
            assert 0 <= x0 < x1 <= p.width + 1 and 0 <= y0 < y1 <= p.height + 1


def test_scanned_blocks_sit_where_digital_ones_do(result):
    """OCR text is placed at the original glyph positions, so bboxes line up."""
    digital = page(result, "digital text").blocks[0].bbox
    scanned = page(result, "scanned text").blocks[0].bbox
    assert np.allclose(digital, scanned, atol=3)


# --- safety -----------------------------------------------------------------------


def test_source_file_is_not_modified(tmp_path, mixed_pdf, settings):
    path = tmp_path / "in.pdf"
    path.write_bytes(mixed_pdf)
    process_pdf(path, settings)
    assert path.read_bytes() == mixed_pdf


def test_text_drawn_as_rectangles_is_recovered_by_ocr(settings):
    with pymupdf.open() as d:
        draw_text_as_rectangles(d.new_page(), ["Invoice DE52660", "Invoice total USD 866.38"])
        result = process_pdf(d.tobytes(), settings)
    p = result.pages[0]
    assert p.method == ExtractionMethod.OCR
    md = _norm(p.markdown)
    assert "de52660" in md and "866.38" in md
    assert p.words  # positions available for source highlighting


def test_ocr_keeps_the_page_image(settings):
    with pymupdf.open() as d:
        add_scanned(d)
        before = d[0].get_pixmap(dpi=40).samples
        create_engine(settings.ocr).ocr_page(d[0])
        assert d[0].get_pixmap(dpi=40).samples == before


def test_non_pdf_is_rejected(tmp_path, settings):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="not a PDF"):
        process_pdf(path, settings)


def test_one_classification_per_page_required(settings):
    with pymupdf.open() as d:
        add_digital(d)
        add_digital(d)
        cls = classify_document(d, settings.classifier)[:1]
        with pytest.raises(ValueError):
            extract_pages(d, cls, create_engine(settings.ocr), settings.extraction)


def test_html_table_format(settings):
    with pymupdf.open() as d:
        add_digital_invoice(d)
        s = settings.model_copy(
            update={"extraction": settings.extraction.model_copy(update={"table_format": "html"})}
        )
        md = process_pdf(d.tobytes(), s).pages[0].markdown
    assert "<table>" in md and "<td>Steel bolts</td>" in md


# --- engine setup -----------------------------------------------------------------


def test_missing_tessdata_dir_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        TesseractEngine(tmp_path / "nope")


def test_missing_language_is_reported(settings):
    with pytest.raises(FileNotFoundError, match="xyz"):
        create_engine(OcrSettings(tessdata_path=settings.ocr.tessdata_path, language="eng+xyz"))


# --- preprocessing (no Tesseract needed) --------------------------------------------


def _gray(a: np.ndarray, dpi: int = 72) -> pymupdf.Pixmap:
    pix = pymupdf.Pixmap(pymupdf.csGRAY, a.shape[1], a.shape[0], a.astype(np.uint8).tobytes(), False)
    pix.set_dpi(dpi, dpi)
    return pix


def _array(pix: pymupdf.Pixmap) -> np.ndarray:
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def test_ruling_lines_are_removed_but_short_strokes_kept():
    a = np.full((100, 200), 255)
    a[50, 10:190] = 0  # long horizontal rule (180 px)
    a[5:95, 100] = 0  # long vertical rule (90 px)
    a[20:30, 20] = 0  # short glyph stroke (10 px)
    out = _array(remove_ruling_lines(_gray(a), min_length_pt=40))
    assert (out[50, 10:190] == 255).all()
    assert (out[5:95, 100] == 255).all()
    assert (out[20:30, 20] == 0).all()


def test_to_rgb_converts_gray():
    rgb = to_rgb(_gray(np.full((10, 10), 128), dpi=300))
    assert rgb.colorspace.n == 3 and not rgb.alpha and rgb.xres == 300
