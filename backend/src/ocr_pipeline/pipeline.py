"""Pipeline entry points: PDF -> per-page Markdown -> LLM-extracted data."""

from __future__ import annotations

from pathlib import Path

import pymupdf
from pydantic import BaseModel

from ocr_pipeline.classify import classify_document
from ocr_pipeline.config import Settings, load_settings
from ocr_pipeline.extract import OcrEngine, create_engine, extract_pages
from ocr_pipeline.linking import link_sources
from ocr_pipeline.llm import LLMProvider, create_provider, extract
from ocr_pipeline.models.document import ExtractedDocument
from ocr_pipeline.models.extraction import ExtractionResult


def _open(source: str | Path | bytes) -> tuple[pymupdf.Document, str]:
    if isinstance(source, bytes):
        doc, name = pymupdf.open(stream=source, filetype="pdf"), "<bytes>"
    else:
        doc, name = pymupdf.open(source), str(source)
    if not doc.is_pdf:
        doc.close()
        raise ValueError(f"not a PDF: {name}")
    return doc, name


def process_pdf(
    source: str | Path | bytes,
    settings: Settings | None = None,
    engine: OcrEngine | None = None,
) -> ExtractedDocument:
    """Classify every page, OCR the ones that need it, and extract Markdown.

    The source file is never modified; OCR happens on an in-memory copy.
    Pass ``engine`` to reuse one across many PDFs (batch processing).
    """
    settings = settings or load_settings()
    engine = engine or create_engine(settings.ocr)

    doc, name = _open(source)
    with doc:
        classifications = classify_document(doc, settings.classifier)
        pages = extract_pages(doc, classifications, engine, settings.extraction)
        return ExtractedDocument(source=name, page_count=doc.page_count, pages=pages)


def extract_from_pdf(
    source: str | Path | bytes,
    schema: type[BaseModel] | dict | None = None,
    instructions: str | None = None,
    settings: Settings | None = None,
    engine: OcrEngine | None = None,
    provider: LLMProvider | None = None,
) -> tuple[ExtractedDocument, ExtractionResult]:
    """Full pipeline: PDF -> per-page Markdown -> LLM-extracted data with sources."""
    settings = settings or load_settings()
    document = process_pdf(source, settings, engine)
    provider = provider or create_provider(settings.llm)
    result = extract(document, provider, schema, instructions, settings.llm.max_retries)
    return document, link_sources(document, result, settings.linking)
