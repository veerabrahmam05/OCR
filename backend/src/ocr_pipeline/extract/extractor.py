"""Turn a classified document into per-page Markdown with layout blocks."""

from __future__ import annotations

import pymupdf
import pymupdf4llm
from pymupdf4llm.ocr import OCRMode

from ocr_pipeline.config import ExtractionSettings
from ocr_pipeline.extract.ocr_engine import OcrEngine
from ocr_pipeline.models.document import (
    ContentBlock,
    ExtractionMethod,
    PageClassification,
    PageContent,
    PageType,
    Word,
)


def _method(c: PageClassification) -> ExtractionMethod:
    if c.needs_ocr:
        return ExtractionMethod.OCR
    if c.page_type == PageType.EMPTY:
        return ExtractionMethod.NONE
    return ExtractionMethod.TEXT_LAYER


def extract_pages(
    doc: pymupdf.Document,
    classifications: list[PageClassification],
    engine: OcrEngine,
    settings: ExtractionSettings,
) -> list[PageContent]:
    """OCR the pages that need it (mutating ``doc`` in memory), then parse all pages once.

    We OCR ourselves instead of letting pymupdf4llm decide: its per-page decision
    skips text-only pages and keeps "valid-looking" text, which would override
    our user-controlled quality threshold. Parsing everything in one call keeps
    header levels consistent across digital and OCR pages.
    """
    if len(classifications) != doc.page_count:
        raise ValueError("need exactly one classification per page")

    for c in classifications:
        if c.needs_ocr:
            engine.ocr_page(doc[c.page_number - 1])

    md_kwargs = {}
    if settings.table_format == "html":
        md_kwargs["table_output"] = "html"
    chunks = pymupdf4llm.to_markdown(
        doc,
        page_chunks=True,
        use_ocr=OCRMode.NEVER,  # OCR already done above
        force_text=True,  # also extract text sitting inside picture regions
        header=settings.include_headers,
        footer=settings.include_footers,
        show_progress=False,
        **md_kwargs,
    )

    pages: list[PageContent] = []
    for c, chunk in zip(classifications, chunks, strict=True):
        markdown: str = chunk["text"] or ""
        blocks = [
            ContentBlock(
                kind=b["class"],
                bbox=tuple(float(v) for v in b["bbox"]),
                start=b["pos"][0],
                end=b["pos"][1],
                text=markdown[b["pos"][0] : b["pos"][1]],
            )
            for b in chunk["page_boxes"] or []
            if b["pos"][1] > b["pos"][0]  # skip boxes that rendered no text
        ]
        # pymupdf4llm removed page rotation in memory, so words and blocks share
        # one (unrotated) coordinate system. Invisible OCR text is included.
        page = doc[c.page_number - 1]
        method = _method(c)
        pages.append(
            PageContent(
                page_number=c.page_number,
                width=page.rect.width,
                height=page.rect.height,
                method=method,
                ocr_engine=engine.name if method == ExtractionMethod.OCR else None,
                markdown=markdown,
                blocks=blocks,
                words=_words(page),
                classification=c,
            )
        )
    return pages


def _words(page: pymupdf.Page) -> list[Word]:
    return [
        Word(text=w[4], bbox=(w[0], w[1], w[2], w[3]), line=w[5] * 10_000 + w[6])
        for w in page.get_text("words", sort=True)
        if w[4].strip()
    ]
