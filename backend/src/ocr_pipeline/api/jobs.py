"""In-memory job store and background runner.

Jobs live only in this process: a restart loses them. PyMuPDF is not thread-safe,
so every PDF operation (reading, OCR, rendering) holds ``PDF_LOCK``; LLM calls run
outside it so several jobs can wait on the network at once.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from ocr_pipeline.api.options import JobOptions, apply_options
from ocr_pipeline.config import LLMSettings, Settings
from ocr_pipeline.extract import OcrEngine, create_engine
from ocr_pipeline.linking import link_sources
from ocr_pipeline.llm import LLMProvider, create_provider, extract
from ocr_pipeline.models.document import ExtractedDocument
from ocr_pipeline.models.extraction import ExtractionResult
from ocr_pipeline.pipeline import read_pdf

log = logging.getLogger(__name__)

PDF_LOCK = threading.RLock()

ProviderFactory = Callable[[LLMSettings], LLMProvider]


class JobStatus(StrEnum):
    QUEUED = "queued"
    CLASSIFYING = "classifying"
    EXTRACTING_TEXT = "extracting_text"  # OCR + layout + Markdown
    EXTRACTING_DATA = "extracting_data"  # LLM
    LINKING = "linking"
    DONE = "done"
    FAILED = "failed"

    @property
    def finished(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED)


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Job:
    id: str
    filename: str
    options: JobOptions
    settings: Settings
    pdf: bytes | None  # original upload; released once text extraction is done
    status: JobStatus = JobStatus.QUEUED
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    document: ExtractedDocument | None = None
    result: ExtractionResult | None = None
    processed_pdf: bytes | None = None  # OCR'd, unrotated copy used for page images

    def set_status(self, status: JobStatus | str) -> None:
        self.status = JobStatus(status)
        self.updated_at = _now()


class JobStoreFull(RuntimeError):
    pass


class JobManager:
    def __init__(
        self,
        settings: Settings,
        provider_factory: ProviderFactory = create_provider,
        engine: OcrEngine | None = None,
    ):
        self.settings = settings
        self.provider_factory = provider_factory
        self._engine = engine
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=settings.api.workers, thread_name_prefix="job")

    # --- store -----------------------------------------------------------------------

    def submit(self, filename: str, pdf: bytes, options: JobOptions) -> Job:
        job_settings = apply_options(self.settings, options)
        if options.extract_data:
            # Fail fast on configuration problems (e.g. no Ollama model, no credentials).
            try:
                self.provider_factory(job_settings.llm)
            except Exception as e:
                raise ValueError(f"LLM provider {job_settings.llm.provider!r} is not usable: {e}") from e
        job = Job(id=uuid.uuid4().hex, filename=filename, options=options, settings=job_settings, pdf=pdf)
        with self._lock:
            self._evict_if_full()
            self._jobs[job.id] = job
        self._pool.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if not job.status.finished:
                raise RuntimeError("job is still running")
            del self._jobs[job_id]
            return True

    def _evict_if_full(self) -> None:
        while len(self._jobs) >= self.settings.api.max_jobs:
            finished = [j for j in self._jobs.values() if j.status.finished]
            if not finished:
                raise JobStoreFull("too many jobs in progress; try again later")
            del self._jobs[min(finished, key=lambda j: j.updated_at).id]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # --- runner ----------------------------------------------------------------------

    def _engine_for(self, settings: Settings) -> OcrEngine:
        if self._engine is not None and settings.ocr == self.settings.ocr:
            return self._engine
        engine = create_engine(settings.ocr)
        if settings.ocr == self.settings.ocr:
            self._engine = engine
        return engine

    def _run(self, job: Job) -> None:
        try:
            with PDF_LOCK:
                document, processed = read_pdf(
                    job.pdf,
                    job.settings,
                    self._engine_for(job.settings),
                    on_stage=job.set_status,
                    keep_processed_pdf=True,
                )
            job.document = document.model_copy(update={"source": job.filename})
            job.processed_pdf = processed
            job.pdf = None

            if job.options.extract_data:
                job.set_status(JobStatus.EXTRACTING_DATA)
                provider = self.provider_factory(job.settings.llm)
                result = extract(
                    job.document,
                    provider,
                    job.options.target_schema(),
                    job.options.instructions,
                    job.settings.llm.max_retries,
                )
                job.set_status(JobStatus.LINKING)
                job.result = link_sources(job.document, result, job.settings.linking)
            job.set_status(JobStatus.DONE)
        except Exception as e:  # report any failure on the job instead of losing it
            log.exception("job %s failed", job.id)
            job.error = f"{type(e).__name__}: {e}"
            job.pdf = None
            job.set_status(JobStatus.FAILED)
