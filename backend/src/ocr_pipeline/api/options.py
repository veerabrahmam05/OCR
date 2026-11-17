"""Per-job options a client may send, and how they override the server settings.

Only the fields below can be changed per job. Everything else (paths, API keys,
limits) stays as configured on the server.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ocr_pipeline.config import DateOrder, Settings
from ocr_pipeline.llm.schema import TargetSchema
from ocr_pipeline.schemas import BUILTIN_SCHEMAS

Provider = Literal["anthropic", "openai", "ollama"]


class JobOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # What to extract: a built-in schema, a custom JSON schema, instructions, or a mix.
    schema_name: str | None = None
    json_schema: dict[str, Any] | None = None
    instructions: str | None = Field(None, max_length=20_000)
    # False = only classify/OCR/Markdown, no LLM call (no API key needed).
    extract_data: bool = True

    provider: Provider | None = None
    model: str | None = Field(None, min_length=1, max_length=200)
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None  # Claude only
    quality_threshold: float | None = Field(None, ge=0, le=1)
    table_format: Literal["markdown", "html"] | None = None
    date_order: DateOrder | None = None

    @model_validator(mode="after")
    def _check(self) -> JobOptions:
        if self.schema_name and self.json_schema:
            raise ValueError("give schema_name or json_schema, not both")
        if self.schema_name and self.schema_name not in BUILTIN_SCHEMAS:
            raise ValueError(
                f"unknown schema_name {self.schema_name!r}; available: {', '.join(BUILTIN_SCHEMAS)}"
            )
        if self.json_schema is not None:
            TargetSchema(self.json_schema)  # raises SchemaError (a ValueError) if unusable
        if self.extract_data and not (
            self.schema_name or self.json_schema or (self.instructions or "").strip()
        ):
            raise ValueError("give schema_name, json_schema or instructions (or set extract_data=false)")
        return self

    def target_schema(self):
        if self.schema_name:
            return BUILTIN_SCHEMAS[self.schema_name]
        return self.json_schema


def apply_options(settings: Settings, opts: JobOptions) -> Settings:
    """A copy of ``settings`` with the job's overrides applied."""
    s = settings.model_copy(deep=True)
    if opts.quality_threshold is not None:
        s.classifier.quality_threshold = opts.quality_threshold
    if opts.table_format is not None:
        s.extraction.table_format = opts.table_format
    if opts.provider is not None:
        s.llm.provider = opts.provider
    if opts.model is not None:
        s.llm.active.model = opts.model
    if opts.effort is not None:
        s.llm.anthropic.effort = opts.effort
    if opts.date_order is not None:
        s.documents.date_order = opts.date_order
    return s
