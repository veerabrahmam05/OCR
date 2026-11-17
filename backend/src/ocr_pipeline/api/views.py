"""Response models (what the API returns, as opposed to internal job state)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from ocr_pipeline.api.jobs import Job, JobStatus
from ocr_pipeline.api.options import JobOptions
from ocr_pipeline.models.document import ExtractedDocument
from ocr_pipeline.models.extraction import ExtractionResult


class JobSummary(BaseModel):
    id: str
    filename: str
    status: JobStatus
    error: str | None
    created_at: datetime
    updated_at: datetime
    page_count: int | None

    @classmethod
    def of(cls, job: Job) -> JobSummary:
        return cls(
            id=job.id,
            filename=job.filename,
            status=job.status,
            error=job.error,
            created_at=job.created_at,
            updated_at=job.updated_at,
            page_count=job.document.page_count if job.document else None,
        )


class JobDetail(JobSummary):
    options: JobOptions
    document: ExtractedDocument | None
    result: ExtractionResult | None

    @classmethod
    def of(cls, job: Job, include_words: bool = False) -> JobDetail:
        document = job.document
        if document is not None and not include_words:
            # Word lists are large and only needed for custom highlighting;
            # source boxes are already resolved in result.sources[].verification.
            document = document.model_copy(
                update={"pages": [p.model_copy(update={"words": []}) for p in document.pages]}
            )
        return cls(
            **JobSummary.of(job).model_dump(),
            options=job.options,
            document=document,
            result=job.result,
        )


class ProviderInfo(BaseModel):
    name: str
    default_model: str
    configured: bool  # False e.g. for Ollama without a model set


class SettingsView(BaseModel):
    """Defaults and allowed values the UI needs to build its option controls."""

    providers: list[ProviderInfo]
    default_provider: str
    efforts: list[str]
    default_effort: str
    quality_threshold: float
    table_formats: list[str]
    default_table_format: str
    schemas: list[str]
    date_orders: list[str]
    default_date_order: str
    max_upload_mb: float
    page_image_max_dpi: int


class SchemaInfo(BaseModel):
    name: str
    json_schema: dict[str, Any]
