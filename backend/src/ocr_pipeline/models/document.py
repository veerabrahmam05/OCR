"""Data models shared across pipeline stages."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class PageType(StrEnum):
    DIGITAL = "digital"  # born-digital text layer
    SCANNED = "scanned"  # page image, with or without an existing OCR layer
    EMPTY = "empty"  # no text, no images, no drawn glyphs


class ClassificationReason(StrEnum):
    EMPTY_PAGE = "empty_page"
    NO_TEXT_LAYER = "no_text_layer"
    IMAGE_DOMINANT_SPARSE_TEXT = "image_dominant_sparse_text"
    TEXT_LAYER_OK = "text_layer_ok"
    TEXT_LAYER_LOW_QUALITY = "text_layer_low_quality"
    OCR_LAYER_OK = "ocr_layer_ok"
    OCR_LAYER_LOW_QUALITY = "ocr_layer_low_quality"


class PageSignals(BaseModel):
    """Raw measurements of a page's text layer, used to compute the quality score."""

    char_count: int = Field(ge=0)  # all non-whitespace span text, visible or not
    ocr_char_count: int = Field(ge=0)  # chars in invisible / GlyphLessFont spans
    word_count: int = Field(ge=0)
    bad_char_ratio: float = Field(ge=0, le=1)  # U+FFFD, control, private-use chars
    dictionary_ratio: float | None = Field(default=None, ge=0, le=1)
    image_coverage: float = Field(ge=0, le=1)  # union of image bboxes / page area
    curve_vector_count: int = Field(ge=0)  # non-rectangle drawings (outlined glyphs?)
    has_ocr_layer: bool
    ocr_model_probability: float | None = Field(default=None, ge=0, le=1)

    @property
    def visible_char_count(self) -> int:
        return self.char_count - self.ocr_char_count


class PageClassification(BaseModel):
    page_number: int = Field(ge=1)  # 1-based
    page_type: PageType
    needs_ocr: bool
    reason: ClassificationReason
    quality_score: float | None = Field(default=None, ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    score_components: dict[str, float] = Field(default_factory=dict)
    signals: PageSignals


# --- extraction output -------------------------------------------------------------

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points, top-left origin


class ExtractionMethod(StrEnum):
    TEXT_LAYER = "text_layer"  # existing text layer used as-is (digital or kept OCR layer)
    OCR = "ocr"  # text layer replaced by fresh OCR
    NONE = "none"  # empty page


class ContentBlock(BaseModel):
    """One layout region of a page and where its text sits in the page Markdown."""

    kind: str  # pymupdf4llm layout class: text, title, section-header, table, list-item, ...
    bbox: BBox
    start: int = Field(ge=0)  # char offsets into PageContent.markdown
    end: int = Field(ge=0)
    text: str  # == PageContent.markdown[start:end]


class Word(BaseModel):
    """One word of the page's text layer (digital or OCR), for precise highlighting."""

    text: str
    bbox: BBox
    line: int  # words sharing this id sit on the same text line


class PageContent(BaseModel):
    page_number: int = Field(ge=1)
    width: float
    height: float
    method: ExtractionMethod
    ocr_engine: str | None = None
    markdown: str
    blocks: list[ContentBlock] = Field(default_factory=list)
    words: list[Word] = Field(default_factory=list)
    classification: PageClassification


class ExtractedDocument(BaseModel):
    """Uniform output of the extraction stage, regardless of digital vs. OCR path."""

    source: str
    page_count: int
    pages: list[PageContent]

    @property
    def markdown(self) -> str:
        return "\n\n".join(p.markdown for p in self.pages if p.markdown)
