"""FastAPI app. Run from backend/:  uv run uvicorn ocr_pipeline.api.app:app --reload"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import get_args

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from ocr_pipeline.api.jobs import PDF_LOCK, JobManager, JobStoreFull, ProviderFactory
from ocr_pipeline.api.options import JobOptions
from ocr_pipeline.api.views import JobDetail, JobSummary, ProviderInfo, SchemaInfo, SettingsView
from ocr_pipeline.config import AnthropicSettings, ExtractionSettings, Settings, load_settings
from ocr_pipeline.extract import OcrEngine
from ocr_pipeline.llm import create_provider
from ocr_pipeline.schemas import BUILTIN_SCHEMAS

def create_app(
    settings: Settings | None = None,
    provider_factory: ProviderFactory = create_provider,
    engine: OcrEngine | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    manager = JobManager(settings, provider_factory, engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(title="OCR pipeline", version="0.1.0", lifespan=lifespan)
    app.state.manager = manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    def get_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        return job

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/settings", response_model=SettingsView)
    def get_settings() -> SettingsView:
        llm = settings.llm
        return SettingsView(
            providers=[
                ProviderInfo(name=name, default_model=getattr(llm, name).model, configured=bool(getattr(llm, name).model))
                for name in ("anthropic", "openai", "ollama")
            ],
            default_provider=llm.provider,
            efforts=list(get_args(AnthropicSettings.model_fields["effort"].annotation)),
            default_effort=llm.anthropic.effort,
            quality_threshold=settings.classifier.quality_threshold,
            table_formats=list(get_args(ExtractionSettings.model_fields["table_format"].annotation)),
            default_table_format=settings.extraction.table_format,
            schemas=list(BUILTIN_SCHEMAS),
            max_upload_mb=settings.api.max_upload_mb,
            page_image_max_dpi=settings.api.page_image_max_dpi,
        )

    @app.get("/schemas", response_model=list[SchemaInfo])
    def list_schemas() -> list[SchemaInfo]:
        return [SchemaInfo(name=n, json_schema=m.model_json_schema()) for n, m in BUILTIN_SCHEMAS.items()]

    @app.post("/jobs", response_model=JobSummary, status_code=status.HTTP_202_ACCEPTED)
    async def create_job(
        file: UploadFile = File(..., description="PDF to process"),
        options: str = Form("{}", description="JobOptions as a JSON string"),
    ) -> JobSummary:
        try:
            opts = JobOptions.model_validate(json.loads(options))
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            if isinstance(e, ValidationError):
                # Only JSON-safe parts: ctx can hold the raw exception object.
                detail = [{"loc": err["loc"], "msg": err["msg"], "type": err["type"]} for err in e.errors()]
            else:
                detail = f"options is not valid JSON: {e}"
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail) from e

        limit = int(settings.api.max_upload_mb * 1024 * 1024)
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE, f"file exceeds {settings.api.max_upload_mb} MB"
            )
        if b"%PDF-" not in data[:1024]:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "file is not a PDF")

        try:
            job = manager.submit(file.filename or "upload.pdf", data, opts)
        except JobStoreFull as e:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
        except ValueError as e:  # provider misconfiguration, e.g. no Ollama model set
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
        return JobSummary.of(job)

    @app.get("/jobs", response_model=list[JobSummary])
    def list_jobs() -> list[JobSummary]:
        return [JobSummary.of(j) for j in manager.list()]

    @app.get("/jobs/{job_id}", response_model=JobDetail)
    def job_detail(job_id: str, include_words: bool = False) -> JobDetail:
        return JobDetail.of(get_job(job_id), include_words)

    @app.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_job(job_id: str) -> Response:
        get_job(job_id)
        try:
            manager.delete(job_id)
        except RuntimeError as e:
            raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get(
        "/jobs/{job_id}/pages/{page_number}/image",
        response_class=Response,
        responses={200: {"content": {"image/png": {}}}},
    )
    def page_image(job_id: str, page_number: int, dpi: int = Query(110, ge=36)) -> Response:
        """PNG of a page, in the same coordinate system as every bbox in the job.

        Box coordinates are PDF points (1/72 inch): multiply by dpi/72 for pixels.
        """
        job = get_job(job_id)
        # Available once text extraction finished, even if the LLM step later failed.
        if job.processed_pdf is None or job.document is None:
            raise HTTPException(status.HTTP_409_CONFLICT, f"page images not ready (status: {job.status})")
        if not 1 <= page_number <= job.document.page_count:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "page not found")
        dpi = min(dpi, settings.api.page_image_max_dpi)
        with PDF_LOCK, pymupdf.open(stream=job.processed_pdf, filetype="pdf") as doc:
            png = doc[page_number - 1].get_pixmap(dpi=dpi).tobytes("png")
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    return app


def __getattr__(name: str):
    # `uvicorn ocr_pipeline.api.app:app` builds the app lazily, so importing this
    # module (e.g. in tests) does not load config or start a worker pool.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
