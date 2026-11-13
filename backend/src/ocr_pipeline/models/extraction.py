"""Models for the LLM extraction stage."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ocr_pipeline.models.document import BBox


class SourceStatus(StrEnum):
    VERIFIED = "verified"  # quote found verbatim (after normalisation) on the cited page
    FUZZY = "fuzzy"  # close match on the cited page (OCR noise, small paraphrase)
    WRONG_PAGE = "wrong_page"  # quote found, but on a different page than cited
    NOT_FOUND = "not_found"  # quote not found anywhere: likely invented
    UNKNOWN_FIELD = "unknown_field"  # cited field path does not exist in the data


class SourceVerification(BaseModel):
    status: SourceStatus
    score: float = Field(ge=0, le=1)  # 1.0 exact; similarity for fuzzy matches
    page: int | None = None  # page where the quote was actually found
    matched_text: str | None = None  # page text that matched the quote
    boxes: list[BBox] = Field(default_factory=list)  # one rectangle per text line
    value: Any = None  # the extracted value the source refers to
    # Does the value itself appear in the matched text? None = cannot tell (e.g. booleans).
    value_found: bool | None = None


class SourceRef(BaseModel):
    """Where an extracted value came from, as reported by the LLM."""

    field: str  # JSON path, e.g. "invoice_number", "line_items[2].amount"
    page: int = Field(ge=1)
    quote: str  # verbatim excerpt of the page text containing the value
    verification: SourceVerification | None = None  # filled by ocr_pipeline.linking


class FieldConflict(BaseModel):
    """Different chunks of a long document returned different values for one field."""

    field: str
    values: list[Any]
    pages: list[list[int]]  # page numbers of the chunk each value came from


class ExtractionResult(BaseModel):
    data: dict[str, Any]
    sources: list[SourceRef] = Field(default_factory=list)
    # Non-null values with no source at all (filled by ocr_pipeline.linking).
    unsourced_fields: list[str] = Field(default_factory=list)
    conflicts: list[FieldConflict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provider: str
    model: str
    chunks: int = Field(ge=1)
    input_tokens: int = 0
    output_tokens: int = 0
