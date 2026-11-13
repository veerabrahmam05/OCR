"""OCR engines.

An engine replaces a page's text layer with OCR text, in place, on an in-memory
document. pymupdf4llm then reads that page like any other text page, so digital
and OCR pages flow through the same layout/Markdown code.

The ``ocr_page`` contract mirrors pymupdf4llm's ``ocr_function`` plugin hook, so
engines can be handed to pymupdf4llm directly later if we choose to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pymupdf

from ocr_pipeline.config import OcrSettings
from ocr_pipeline.extract.preprocess import remove_ruling_lines, to_rgb


class OcrEngine(ABC):
    name: str

    @abstractmethod
    def ocr_page(self, page: pymupdf.Page) -> None:
        """Replace ``page``'s text layer with OCR text. Images and drawings are kept."""


class TesseractEngine(OcrEngine):
    """Tesseract via PyMuPDF's built-in binding (the same call pymupdf4llm uses)."""

    name = "tesseract"

    def __init__(
        self,
        tessdata: str | Path | None,
        language: str = "eng",
        dpi: int = 300,
        remove_lines: bool = True,
        line_min_length_pt: float = 40.0,
    ):
        self.tessdata = self._resolve_tessdata(tessdata)
        self.language = language
        self.dpi = dpi
        self.remove_lines = remove_lines
        self.line_min_length_pt = line_min_length_pt
        missing = [
            lang
            for lang in language.split("+")
            if not (self.tessdata / f"{lang}.traineddata").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Tesseract language data not found in {self.tessdata}: {', '.join(missing)}"
            )

    @staticmethod
    def _resolve_tessdata(tessdata: str | Path | None) -> Path:
        if tessdata is not None:
            path = Path(tessdata)
        else:
            try:
                path = Path(pymupdf.get_tessdata())
            except RuntimeError as e:
                raise FileNotFoundError(
                    "Tesseract tessdata not found. Set [ocr].tessdata_path in the config "
                    "or the TESSDATA_PREFIX environment variable."
                ) from e
        if not path.is_dir():
            raise FileNotFoundError(f"tessdata directory does not exist: {path}")
        return path

    def ocr_page(self, page: pymupdf.Page) -> None:
        page.remove_rotation()

        # 1. Render first, while every glyph is still visible. Broken-encoding text
        #    looks fine on screen; only its extracted text is garbage.
        pix = page.get_pixmap(dpi=self.dpi)

        # 2. Drop the old text layer (garbage or stale OCR). Keep images and vectors.
        page.add_redact_annot(page.rect)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )

        # 3. Clean the OCR copy only (the page keeps its lines), OCR it into a
        #    one-page PDF (image + invisible text), keep only the text and lay it
        #    over the original page.
        if self.remove_lines:
            pix = remove_ruling_lines(pix, self.line_min_length_pt)
        data = to_rgb(pix).pdfocr_tobytes(language=self.language, tessdata=str(self.tessdata))
        with pymupdf.open("pdf", data) as ocr_doc:
            ocr_page = ocr_doc[0]
            ocr_page.add_redact_annot(ocr_page.rect)
            ocr_page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_REMOVE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                text=pymupdf.PDF_REDACT_TEXT_NONE,
            )
            page.show_pdf_page(page.rect, ocr_doc, 0)


def create_engine(settings: OcrSettings) -> OcrEngine:
    if settings.engine == "tesseract":
        return TesseractEngine(
            settings.tessdata_path,
            settings.language,
            settings.dpi,
            remove_lines=settings.remove_table_lines,
            line_min_length_pt=settings.line_min_length_pt,
        )
    raise ValueError(f"unknown OCR engine: {settings.engine}")
