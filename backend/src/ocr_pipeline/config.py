"""Pipeline settings.

The Pydantic defaults below are the source of truth. A TOML file can override any
of them; it is looked up in this order:

1. the ``path`` argument of :func:`load_settings`
2. the ``OCR_PIPELINE_CONFIG`` environment variable
3. ``config/default.toml`` relative to the current working directory

Per-request overrides (e.g. a threshold chosen in the UI) are applied with
``settings.model_copy(update=...)`` and never written back to the file.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

CONFIG_ENV_VAR = "OCR_PIPELINE_CONFIG"
DEFAULT_CONFIG_PATH = Path("config") / "default.toml"


class QualityWeights(BaseModel):
    """Relative weights of the text-layer quality components (renormalised at use)."""

    dictionary: float = Field(0.5, ge=0)
    clean_chars: float = Field(0.3, ge=0)
    image_text: float = Field(0.2, ge=0)

    @model_validator(mode="after")
    def _at_least_one_positive(self) -> QualityWeights:
        if self.dictionary + self.clean_chars + self.image_text <= 0:
            raise ValueError("at least one quality weight must be positive")
        return self


class ClassifierSettings(BaseModel):
    """Controls how each page is classified as digital / scanned / empty."""

    # Pages whose text layer scores below this are (re-)OCRed. Exposed in the UI.
    quality_threshold: float = Field(0.6, ge=0, le=1)
    # Image covering at least this fraction of the page, with fewer visible
    # characters than image_dominant_max_chars, means "scan with a small stamp".
    image_dominant_coverage: float = Field(0.6, ge=0, le=1)
    image_dominant_max_chars: int = Field(200, ge=0)
    # A page with nothing extractable is "empty" only if at most this share of it is
    # printed on when rendered; above it, the page is OCRed (content is hidden from
    # extraction, e.g. in annotations). 0.002 = 0.2% of pixels.
    empty_max_ink: float = Field(0.002, ge=0, le=1)
    # Fraction of characters that must be invisible/OCR text to call it an OCR layer.
    ocr_layer_min_fraction: float = Field(0.8, ge=0, le=1)
    # Below this many words the dictionary ratio is too noisy and is skipped.
    min_words_for_dictionary: int = Field(5, ge=1)
    # clean_chars component = 1 - bad_char_ratio * penalty (20% bad chars -> 0).
    bad_char_penalty: float = Field(5.0, gt=0)
    # Hard gate: more bad chars than this means a broken text layer, whatever the score.
    max_bad_char_ratio: float = Field(0.1, ge=0, le=1)
    # Use pymupdf4llm's ONNX OCR-decision model as one score component.
    use_ocr_model: bool = True
    weights: QualityWeights = Field(default_factory=QualityWeights)


class OcrSettings(BaseModel):
    """OCR engine options."""

    engine: Literal["tesseract"] = "tesseract"
    # None -> rely on the TESSDATA_PREFIX environment variable.
    tessdata_path: Path | None = None
    language: str = "eng"  # Tesseract syntax; "eng+deu" for several
    dpi: int = Field(300, ge=72, le=1200)
    # Erase table rulings from the OCR image (not the page): Tesseract drops ruled cells.
    remove_table_lines: bool = True
    line_min_length_pt: float = Field(40.0, gt=0)  # shorter dark runs are left alone


class ExtractionSettings(BaseModel):
    """Options passed to pymupdf4llm when building Markdown."""

    # Page headers/footers often carry invoice numbers, dates, page refs: keep by default.
    include_headers: bool = True
    include_footers: bool = True
    # "html" keeps merged cells (rowspan/colspan); "markdown" is more compact.
    table_format: Literal["markdown", "html"] = "markdown"


class ProviderSettings(BaseModel):
    """Per-provider LLM options. API keys are never stored here: they come from env / .env."""

    model: str = ""
    # Chunking budget: documents larger than this (estimated tokens) are split by pages.
    max_input_tokens: int = Field(100_000, ge=1_000)
    max_output_tokens: int = Field(16_000, ge=256)


class AnthropicSettings(ProviderSettings):
    model: str = "claude-opus-5-5"
    max_input_tokens: int = Field(150_000, ge=1_000)
    max_output_tokens: int = Field(64_000, ge=256)  # streamed, so a high cap is safe
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"


class OpenAISettings(ProviderSettings):
    model: str = "gpt-5"
    base_url: str | None = None  # for OpenAI-compatible gateways; None = api.openai.com


class OllamaSettings(ProviderSettings):
    model: str = ""  # must be set, e.g. "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    # Ollama silently truncates prompts longer than num_ctx, so the input budget
    # must leave room for the prompt overhead and the output.
    num_ctx: int = Field(16_384, ge=2_048)
    max_input_tokens: int = Field(10_000, ge=1_000)
    max_output_tokens: int = Field(4_096, ge=256)
    # Reasoning before answering ("thinking" models such as qwen3.x). Off by default:
    # on CPU it multiplies response time. None = leave it to the model's default.
    think: bool | None = False

    @model_validator(mode="after")
    def _fits_context(self) -> OllamaSettings:
        if self.max_input_tokens + self.max_output_tokens > self.num_ctx:
            raise ValueError("ollama: max_input_tokens + max_output_tokens must not exceed num_ctx")
        return self


class LLMSettings(BaseModel):
    provider: Literal["anthropic", "openai", "ollama"] = "anthropic"
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    # One retry with the validation error fed back to the model.
    max_retries: int = Field(1, ge=0, le=3)

    @property
    def active(self) -> ProviderSettings:
        return getattr(self, self.provider)


class LinkingSettings(BaseModel):
    """Verification of LLM-reported sources against the page text."""

    # Minimum similarity (0-1) for a quote that is not an exact match, e.g. OCR noise.
    fuzzy_threshold: float = Field(0.85, ge=0.5, le=1)
    # If the quote is not on the cited page, look for it on the other pages.
    search_other_pages: bool = True


DateOrder = Literal["auto", "DMY", "MDY", "YMD"]


class DocumentSettings(BaseModel):
    """Conventions of the documents being processed."""

    # How numeric dates are written. "DMY": 12/11/2026 is 12 November; "MDY": 11 December.
    # Told to the LLM and used when checking extracted dates against the page.
    # "auto" gives no hint and accepts either reading.
    date_order: DateOrder = "auto"


class ApiSettings(BaseModel):
    """HTTP API options. Job results live in memory only and are lost on restart."""

    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    max_upload_mb: float = Field(50, gt=0)
    # Finished jobs beyond this are evicted, oldest first.
    max_jobs: int = Field(100, ge=1)
    # Concurrent jobs. PDF work is serialised by a lock (PyMuPDF is not thread-safe);
    # LLM calls overlap.
    workers: int = Field(2, ge=1, le=16)
    page_image_max_dpi: int = Field(200, ge=36, le=600)


class Settings(BaseModel):
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    linking: LinkingSettings = Field(default_factory=LinkingSettings)
    documents: DocumentSettings = Field(default_factory=DocumentSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings, applying the first TOML override file found (see module doc)."""
    if path is None:
        env_path = os.environ.get(CONFIG_ENV_VAR)
        if env_path:
            path = env_path
        elif DEFAULT_CONFIG_PATH.is_file():
            path = DEFAULT_CONFIG_PATH
        else:
            return Settings()

    with open(path, "rb") as f:
        return Settings.model_validate(tomllib.load(f))
