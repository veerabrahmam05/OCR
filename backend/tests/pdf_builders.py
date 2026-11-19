"""Build synthetic test pages in memory: digital, scanned, garbage, tables."""

from __future__ import annotations

import random

import pymupdf

ENGLISH = [
    "This agreement is made between the buyer and the seller on the date below.",
    "The seller agrees to deliver the goods to the warehouse within thirty days.",
    "Payment will be made in full after the goods are received and inspected.",
    "Any dispute arising from this contract shall be settled by arbitration.",
    "The total amount payable includes all taxes, shipping and handling costs.",
    "Please contact our office if you have any questions about this invoice.",
    "Both parties confirm that they have read and understood these terms.",
    "This document remains valid until it is replaced by a new written agreement.",
]

TABLE_ROWS = [
    ["Item", "Qty", "Unit price", "Amount"],
    ["Steel bolts", "200", "0.50", "100.00"],
    ["Copper wire", "15", "12.00", "180.00"],
    ["Safety gloves", "40", "3.25", "130.00"],
]
_TABLE_COLS = [60, 230, 310, 420, 530]  # column x boundaries


def gibberish(lines: int = 8, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    consonants = "bcdfghjklmnpqrstvwxz"

    def word() -> str:
        return "".join(rng.choice(consonants) for _ in range(rng.randint(3, 8)))

    return [" ".join(word() for _ in range(10)) for _ in range(lines)]


def write_lines(page: pymupdf.Page, lines: list[str], render_mode: int = 0) -> None:
    y = 72
    for line in lines:
        page.insert_text((50, y), line, fontsize=10, render_mode=render_mode)
        y += 16


def draw_invoice(page: pymupdf.Page, ruled: bool = True) -> None:
    """An invoice with a 4x4 item table, optionally with ruling lines."""
    page.insert_text((60, 70), "Invoice INV-2026-0042", fontsize=16, fontname="hebo")
    page.insert_text((60, 95), "Billed to: Acme Corporation, 12 Market Street", fontsize=10)
    top, rh = 120, 22
    for r, row in enumerate(TABLE_ROWS):
        for c, cell in enumerate(row):
            page.insert_text(
                (_TABLE_COLS[c] + 6, top + r * rh + 15),
                cell,
                fontsize=10,
                fontname="hebo" if r == 0 else "helv",
            )
    if ruled:
        bottom = top + len(TABLE_ROWS) * rh
        for r in range(len(TABLE_ROWS) + 1):
            page.draw_line((_TABLE_COLS[0], top + r * rh), (_TABLE_COLS[-1], top + r * rh))
        for x in _TABLE_COLS:
            page.draw_line((x, top), (x, bottom))
    page.insert_text((60, top + len(TABLE_ROWS) * rh + 30), "Total due: 410.00 USD", fontsize=10)


def draw_text_as_rectangles(page: pymupdf.Page, lines: list[str], fontsize: float = 16) -> None:
    """Draw text as runs of tiny filled rectangles, one drawing per run.

    Mimics generators that rasterise a page into vector rectangles: it prints like
    text, but has no text, no images and no curves for extraction to find.
    """
    import numpy as np

    dpi = 100
    with pymupdf.open() as tmp:
        t = tmp.new_page(width=page.rect.width, height=page.rect.height)
        y = 60 + fontsize
        for line in lines:
            t.insert_text((50, y), line, fontsize=fontsize)
            y += fontsize * 1.6
        pix = t.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    dark = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width) < 128
    scale = 72 / dpi
    for row in range(pix.height):
        cols = np.flatnonzero(dark[row])
        if not cols.size:
            continue
        # split the dark pixels of this row into contiguous runs
        breaks = np.flatnonzero(np.diff(cols) > 1)
        starts = np.concatenate(([cols[0]], cols[breaks + 1]))
        ends = np.concatenate((cols[breaks], [cols[-1]]))
        for x0, x1 in zip(starts, ends, strict=True):
            r = pymupdf.Rect(x0 * scale, row * scale, (x1 + 1) * scale, (row + 1) * scale)
            page.draw_rect(r, color=None, fill=(0, 0, 0), width=0)


def add_digital(doc: pymupdf.Document, lines: list[str] = ENGLISH) -> None:
    write_lines(doc.new_page(), lines)


def add_digital_invoice(doc: pymupdf.Document, ruled: bool = True) -> None:
    draw_invoice(doc.new_page(), ruled)


def scan_of(draw) -> pymupdf.Pixmap:
    """Render whatever ``draw(page)`` puts on a page to an image, like a scanner."""
    with pymupdf.open() as tmp:
        page = tmp.new_page()
        draw(page)
        return page.get_pixmap(dpi=150)


def add_scanned(
    doc: pymupdf.Document,
    draw=lambda p: write_lines(p, ENGLISH),
    ocr_layer: list[str] | None = None,
    visible_text: str | None = None,
) -> None:
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=scan_of(draw))
    if ocr_layer:
        write_lines(page, ocr_layer, render_mode=3)  # invisible text, like an OCR layer
    if visible_text:
        page.insert_text((50, 800), visible_text, fontsize=8)
