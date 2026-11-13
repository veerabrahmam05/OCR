"""Image clean-up applied to the copy of a page that is sent to OCR.

The page itself is never modified by these functions.
"""

from __future__ import annotations

import numpy as np
import pymupdf

DARK_THRESHOLD = 200  # gray value; < 200 also catches anti-aliased line edges


def _long_runs(dark: np.ndarray, k: int) -> np.ndarray:
    """Mask of pixels lying on a horizontal run of at least ``k`` dark pixels."""
    h, w = dark.shape
    if k > w:
        return np.zeros_like(dark)
    cs = np.zeros((h, w + 1), dtype=np.int32)
    np.cumsum(dark, axis=1, out=cs[:, 1:])
    starts = (cs[:, k:] - cs[:, :-k]) == k  # window [i, i+k) is fully dark
    cover = np.zeros((h, w + 1), dtype=np.int32)
    cover[:, : w - k + 1] += starts
    cover[:, k:] -= starts
    return np.cumsum(cover, axis=1)[:, :w] > 0


def remove_ruling_lines(pix: pymupdf.Pixmap, min_length_pt: float) -> pymupdf.Pixmap:
    """Whiten long horizontal and vertical lines (table rulings, underlines).

    Tesseract's page segmentation breaks on ruled table cells: in testing it
    dropped every data row of a simple ruled invoice table. With the rulings
    removed it read the same table perfectly.
    """
    gray = pymupdf.Pixmap(pymupdf.csGRAY, pix) if pix.colorspace.n != 1 else pix
    a = np.frombuffer(gray.samples, dtype=np.uint8).reshape(gray.height, gray.width).copy()
    k = max(1, round(min_length_pt * pix.xres / 72))
    dark = a < DARK_THRESHOLD
    lines = _long_runs(dark, k) | _long_runs(dark.T, k).T
    a[lines] = 255
    out = pymupdf.Pixmap(pymupdf.csGRAY, gray.width, gray.height, a.tobytes(), False)
    out.set_dpi(pix.xres, pix.yres)
    return out


def to_rgb(pix: pymupdf.Pixmap) -> pymupdf.Pixmap:
    """Tesseract via ``pdfocr_tobytes`` silently returns no text for grayscale input."""
    if pix.colorspace is not None and pix.colorspace.n == 3 and not pix.alpha:
        return pix
    rgb = pymupdf.Pixmap(pymupdf.csRGB, pix)
    if rgb.alpha:
        rgb = pymupdf.Pixmap(rgb, 0)
    rgb.set_dpi(pix.xres, pix.yres)
    return rgb
