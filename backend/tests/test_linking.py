"""Source verification: quote matching, value checks, field paths, page boxes."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pymupdf
import pytest
from llm_fakes import FakeProvider, make_document
from pdf_builders import draw_invoice

from ocr_pipeline.config import LinkingSettings, load_settings
from ocr_pipeline.extract import create_engine
from ocr_pipeline.linking import link_sources
from ocr_pipeline.linking.paths import covers, leaf_paths, resolve
from ocr_pipeline.linking.text import tokens
from ocr_pipeline.linking.values import dates_in, numbers_in, value_in_text
from ocr_pipeline.models import ExtractionResult, SourceRef, SourceStatus, Word
from ocr_pipeline.pipeline import extract_from_pdf

SETTINGS = LinkingSettings()


def words_from_lines(lines: list[str]) -> list[Word]:
    """Fake word boxes: 6pt per character, 20pt per line, starting at (50, 100)."""
    out = []
    for li, line in enumerate(lines):
        x = 50.0
        for w in line.split():
            out.append(Word(text=w, bbox=(x, 100 + 20 * li, x + 6 * len(w), 112 + 20 * li), line=li))
            x += 6 * len(w) + 4
    return out


def doc_with(*pages: list[str]):
    doc = make_document(["\n".join(p) for p in pages])
    for page, lines in zip(doc.pages, pages, strict=True):
        page.words = words_from_lines(lines)
    return doc


def result(data: dict, *sources: tuple[str, int, str]) -> ExtractionResult:
    return ExtractionResult(
        data=data,
        sources=[SourceRef(field=f, page=p, quote=q) for f, p, q in sources],
        provider="fake",
        model="fake",
        chunks=1,
    )


def verify(doc, data, *sources):
    return link_sources(doc, result(data, *sources), SETTINGS)


INVOICE_PAGE = [
    "Invoice INV-2026-0042",
    "Date: 12/11/2026",
    "Steel bolts 200 0.50 100.00",
    "Safetygloves 40 3.25 130.00",
    "Total due: 1,250.00 USD",
]


# --- status ---------------------------------------------------------------------------


def test_exact_quote_is_verified_with_a_box():
    doc = doc_with(INVOICE_PAGE)
    r = verify(doc, {"invoice_number": "INV-2026-0042"}, ("invoice_number", 1, "Invoice INV-2026-0042"))
    v = r.sources[0].verification
    assert v.status == SourceStatus.VERIFIED and v.score == 1.0
    assert v.page == 1 and v.matched_text == "Invoice INV-2026-0042"
    assert v.boxes == [(50.0, 100.0, 50 + 6 * 7 + 4 + 6 * 13, 112.0)]
    assert v.value == "INV-2026-0042" and v.value_found is True


def test_markdown_in_quote_is_ignored():
    doc = doc_with(INVOICE_PAGE)
    r = verify(doc, {"total": 1250.0}, ("total", 1, "**Total due:** 1,250.00 USD"))
    assert r.sources[0].verification.status == SourceStatus.VERIFIED


def test_table_row_quote_matches_across_cells():
    doc = doc_with(INVOICE_PAGE)
    r = verify(doc, {"amount": 100.0}, ("amount", 1, "|Steel bolts|200|0.50|100.00|"))
    assert r.sources[0].verification.status == SourceStatus.VERIFIED


def test_ocr_noise_gives_fuzzy_match():
    doc = doc_with(INVOICE_PAGE)
    r = verify(doc, {"item": "Safety gloves"}, ("item", 1, "Safety gloves 40 3.25 130.00"))
    v = r.sources[0].verification
    assert v.status == SourceStatus.FUZZY and 0.85 <= v.score < 1
    assert v.matched_text == "Safetygloves 40 3.25 130.00"
    assert v.value_found is True  # fuzzy text match on the value too


def test_quote_spanning_lines_gives_one_box_per_line():
    doc = doc_with(INVOICE_PAGE)
    r = verify(doc, {"x": "INV-2026-0042"}, ("x", 1, "Invoice INV-2026-0042 Date: 12/11/2026"))
    boxes = r.sources[0].verification.boxes
    assert len(boxes) == 2 and boxes[0][1] < boxes[1][1]


def test_invented_quote_is_not_found():
    doc = doc_with(INVOICE_PAGE)
    r = verify(doc, {"po": "PO-7781"}, ("po", 1, "Purchase order PO-7781"))
    v = r.sources[0].verification
    assert v.status == SourceStatus.NOT_FOUND and v.boxes == [] and v.value_found is None


def test_quote_on_another_page_is_flagged_with_the_real_page():
    doc = doc_with(["Cover letter"], INVOICE_PAGE)
    r = verify(doc, {"total": 1250.0}, ("total", 1, "Total due: 1,250.00 USD"))
    v = r.sources[0].verification
    assert v.status == SourceStatus.WRONG_PAGE and v.page == 2 and v.boxes


def test_other_pages_not_searched_when_disabled():
    doc = doc_with(["Cover letter"], INVOICE_PAGE)
    r = link_sources(
        doc,
        result({"total": 1250.0}, ("total", 1, "Total due: 1,250.00 USD")),
        LinkingSettings(search_other_pages=False),
    )
    assert r.sources[0].verification.status == SourceStatus.NOT_FOUND


def test_unknown_field_path():
    doc = doc_with(INVOICE_PAGE)
    r = verify(doc, {"total": 1250.0}, ("grand_total", 1, "Total due: 1,250.00 USD"))
    assert r.sources[0].verification.status == SourceStatus.UNKNOWN_FIELD
    assert r.unsourced_fields == ["total"]  # an unknown-field source covers nothing


def test_value_missing_from_its_quote_is_caught():
    doc = doc_with(INVOICE_PAGE)
    # Right place, wrong number: the model misread 1,250.00 as 1520.
    r = verify(doc, {"total": 1520.0}, ("total", 1, "Total due: 1,250.00 USD"))
    v = r.sources[0].verification
    assert v.status == SourceStatus.VERIFIED and v.value_found is False


def test_unsourced_fields_listed():
    doc = doc_with(INVOICE_PAGE)
    data = {
        "invoice_number": "INV-2026-0042",
        "total": 1250.0,
        "notes": None,
        "line_items": [{"description": "Steel bolts", "amount": 100.0}],
    }
    r = verify(doc, data, ("line_items[0]", 1, "Steel bolts 200 0.50 100.00"))
    assert r.unsourced_fields == ["invoice_number", "total"]


# --- helpers --------------------------------------------------------------------------


def test_tokens_strip_markup_and_punctuation():
    assert tokens("|**Total due:**| 1,250.00 USD. <td>x</td> ---") == ["total", "due", "1,250.00", "usd", "x"]


def test_numbers_in_reads_us_and_eu_formats():
    assert {1250.0} <= numbers_in("1,250.00") and {1250.0} <= numbers_in("1.250,00")


@pytest.mark.parametrize(
    "text", ["2026-11-12", "12/11/2026", "11/12/2026", "12.11.26", "12 Nov 2026", "November 12th, 2026"]
)
def test_dates_in_reads_common_formats(text):
    assert date(2026, 11, 12) in dates_in(text)


@pytest.mark.parametrize(
    "value, text, expected",
    [
        (1250.0, "Total: 1,250.00", True),
        (1250, "Total: 1.250,00 EUR", True),
        (125.0, "Total: 1,250.00", False),
        ("2026-11-12", "Dated 12 November 2026", True),
        ("2026-11-13", "Dated 12 November 2026", False),
        ("Acme Corporation", "Billed to: ACME CORPORATION, 12 Market St", True),
        ("Globex", "Billed to: Acme Corporation", False),
        ("280.00", "Total 280.00 USD", True),
        (True, "anything", None),
        (None, "anything", None),
        ({"a": 1}, "anything", None),
    ],
)
def test_value_in_text(value, text, expected):
    assert value_in_text(value, text) is expected


def test_resolve_paths():
    data = {"a": {"b": [{"c": 1}, {"c": 2}]}, "z": None}
    assert resolve(data, "a.b[1].c") == (True, 2)
    assert resolve(data, "z") == (True, None)
    assert resolve(data, "a.b[5].c") == (False, None)
    free = {"fields": [{"name": "total", "value": "280.00"}]}
    assert resolve(free, "total") == (True, "280.00")


def test_leaf_paths_and_covers():
    data = {"a": 1, "b": None, "items": [{"x": 1, "y": None}], "free": []}
    assert leaf_paths(data) == ["a", "items[0].x"]
    assert leaf_paths({"fields": [{"name": "n", "value": "v"}, {"name": "m", "value": None}]}) == ["n"]
    assert covers("items[0]", "items[0].x") and covers("items", "items[0].x")
    assert not covers("item", "items[0].x")


# --- end to end on real pages (digital + OCR) --------------------------------------------

REPO_CONFIG = Path(__file__).parents[1] / "config" / "default.toml"


@pytest.fixture(scope="module")
def settings():
    s = load_settings(REPO_CONFIG)
    try:
        create_engine(s.ocr)
    except FileNotFoundError as e:
        pytest.skip(f"Tesseract not available: {e}")
    return s


@pytest.fixture(scope="module")
def invoice_pdf() -> bytes:
    """Page 1: digital invoice. Page 2: the same invoice as a scan."""
    with pymupdf.open() as d:
        draw_invoice(d.new_page())
        with pymupdf.open() as tmp:
            draw_invoice(tmp.new_page())
            pix = tmp[0].get_pixmap(dpi=150)
        p = d.new_page()
        p.insert_image(p.rect, pixmap=pix)
        return d.tobytes()


@pytest.mark.parametrize("page", [1, 2], ids=["digital", "scanned"])
def test_sources_located_on_real_pages(settings, invoice_pdf, page):
    reply = {
        "data": {"invoice_number": "INV-2026-0042", "total": 410.0, "line_items": []},
        "sources": [
            {"field": "invoice_number", "page": page, "quote": "Invoice INV-2026-0042"},
            {"field": "total", "page": page, "quote": "Total due: 410.00 USD"},
        ],
    }
    from ocr_pipeline.schemas import Invoice

    document, r = extract_from_pdf(
        invoice_pdf, Invoice, settings=settings, provider=FakeProvider([reply])
    )
    assert document.pages[page - 1].words, "page should have word positions"
    by_field = {s.field: s.verification for s in r.sources}

    inv = by_field["invoice_number"]
    assert inv.status == SourceStatus.VERIFIED and inv.value_found
    # draw_invoice puts the title baseline at y=70 (16pt) starting at x=60
    (x0, y0, x1, y1), = inv.boxes
    assert 55 < x0 < 65 and 50 < y0 < 70 and 60 < y1 < 80

    total = by_field["total"]
    assert total.status == SourceStatus.VERIFIED and total.value_found
    (_, ty0, _, ty1), = total.boxes
    # "Total due" is drawn at baseline y = 120 + 4*22 + 30 = 238 (10pt): the box must
    # span that line (OCR boxes are a little tighter than font boxes).
    assert 225 < ty0 < 238 <= ty1 + 3 and ty1 < 245
    assert r.unsourced_fields == []
