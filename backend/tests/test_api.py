"""HTTP API tests with a fake LLM provider (real classification/OCR, no network)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from llm_fakes import FakeProvider
from pdf_builders import add_digital, draw_invoice

from ocr_pipeline.api.app import create_app
from ocr_pipeline.config import load_settings
from ocr_pipeline.extract import create_engine
from ocr_pipeline.llm.providers.base import LLMRefusal

REPO_CONFIG = Path(__file__).parents[1] / "config" / "default.toml"

INVOICE_REPLY = {
    "data": {"invoice_number": "INV-2026-0042", "total": 410.0, "line_items": []},
    "sources": [
        {"field": "invoice_number", "page": 1, "quote": "Invoice INV-2026-0042"},
        {"field": "total", "page": 1, "quote": "Total due: 410.00 USD"},
    ],
}


@pytest.fixture(scope="module")
def base_settings():
    s = load_settings(REPO_CONFIG)
    try:
        engine = create_engine(s.ocr)
    except FileNotFoundError as e:
        pytest.skip(f"Tesseract not available: {e}")
    return s, engine


class Factory:
    """Provider factory that records the LLM settings each job was given."""

    def __init__(self):
        self.replies: list = []
        self.seen = []

    def __call__(self, llm):
        self.seen.append(llm)
        return FakeProvider(self.replies)  # shares the queue: one reply per LLM call


@pytest.fixture
def api(base_settings):
    settings, engine = base_settings
    factory = Factory()
    with TestClient(create_app(settings, factory, engine)) as client:
        yield client, factory


@pytest.fixture(scope="module")
def invoice_pdf() -> bytes:
    with pymupdf.open() as d:
        draw_invoice(d.new_page())
        add_digital(d)
        return d.tobytes()


def submit(client, pdf: bytes, filename="invoice.pdf", **options):
    return client.post(
        "/jobs",
        files={"file": (filename, pdf, "application/pdf")},
        data={"options": json.dumps(options)},
    )


def wait(client, job_id: str, timeout: float = 60) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} still {job['status']}")


# --- metadata ------------------------------------------------------------------------


def test_health(api):
    assert api[0].get("/health").json() == {"status": "ok"}


def test_settings_for_the_ui(api):
    s = api[0].get("/settings").json()
    assert s["default_provider"] == "anthropic"
    assert [p["name"] for p in s["providers"]] == ["anthropic", "openai", "ollama"]
    assert all(p["configured"] == bool(p["default_model"]) for p in s["providers"])
    assert "medium" in s["efforts"] and s["table_formats"] == ["markdown", "html"]
    assert s["schemas"] == ["invoice", "contract"]
    assert 0 < s["quality_threshold"] < 1


def test_schemas(api):
    schemas = {s["name"]: s["json_schema"] for s in api[0].get("/schemas").json()}
    assert "invoice_number" in schemas["invoice"]["properties"]


# --- full job ----------------------------------------------------------------------


def test_invoice_job_end_to_end(api, invoice_pdf):
    client, factory = api
    factory.replies.append(INVOICE_REPLY)
    r = submit(client, invoice_pdf, schema_name="invoice", instructions="Amounts in USD.")
    assert r.status_code == 202
    created = r.json()
    # The worker may already be at any stage by the time we get the response.
    assert created["status"] != "failed" and created["filename"] == "invoice.pdf"

    job = wait(client, created["id"])
    assert job["status"] == "done", job["error"]
    assert job["page_count"] == 2
    assert job["document"]["source"] == "invoice.pdf"
    assert all(p["words"] == [] for p in job["document"]["pages"])  # trimmed by default

    result = job["result"]
    assert result["data"]["invoice_number"] == "INV-2026-0042"
    statuses = {s["field"]: s["verification"]["status"] for s in result["sources"]}
    assert statuses == {"invoice_number": "verified", "total": "verified"}
    assert result["sources"][0]["verification"]["boxes"]

    with_words = client.get(f"/jobs/{created['id']}", params={"include_words": True}).json()
    assert with_words["document"]["pages"][0]["words"]


def test_text_only_job_makes_no_llm_call(api, invoice_pdf):
    client, factory = api
    job = wait(client, submit(client, invoice_pdf, extract_data=False).json()["id"])
    assert job["status"] == "done" and job["result"] is None
    assert "INV-2026-0042" in job["document"]["pages"][0]["markdown"]
    assert factory.seen == []


def test_overrides_reach_the_pipeline(api, invoice_pdf):
    client, factory = api
    factory.replies.append(INVOICE_REPLY)
    r = submit(
        client, invoice_pdf, schema_name="invoice", provider="openai", model="gpt-test",
        quality_threshold=0.42, table_format="html",
    )  # fmt: skip
    job = wait(client, r.json()["id"])
    assert job["status"] == "done", job["error"]
    assert factory.seen[-1].provider == "openai" and factory.seen[-1].openai.model == "gpt-test"
    assert job["document"]["pages"][0]["classification"]["threshold"] == 0.42
    assert "<table>" in job["document"]["pages"][0]["markdown"]


def test_llm_failure_marks_job_failed(api, invoice_pdf):
    client, factory = api
    factory.replies.append(LLMRefusal("declined (category: cyber)"))
    job = wait(client, submit(client, invoice_pdf, instructions="all values").json()["id"])
    assert job["status"] == "failed"
    assert "declined" in job["error"]
    assert job["document"] is not None  # text extraction still available


# --- page images ----------------------------------------------------------------------


def test_page_image(api, invoice_pdf):
    client, _ = api
    job = wait(client, submit(client, invoice_pdf, extract_data=False).json()["id"])
    r = client.get(f"/jobs/{job['id']}/pages/1/image", params={"dpi": 72})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    pix = pymupdf.Pixmap(r.content)
    assert (pix.width, pix.height) == (595, 842)  # A4 at 72 dpi = PDF points

    capped = pymupdf.Pixmap(client.get(f"/jobs/{job['id']}/pages/1/image", params={"dpi": 5000}).content)
    assert capped.width == round(595 * 200 / 72)  # page_image_max_dpi
    assert client.get(f"/jobs/{job['id']}/pages/3/image").status_code == 404


# --- validation -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {},  # nothing to extract
        {"schema_name": "invoice", "json_schema": {"type": "object"}},
        {"schema_name": "receipt"},
        {"instructions": "x", "unexpected": 1},
        {"instructions": "x", "quality_threshold": 2},
        {"json_schema": {"type": "array"}},
    ],
)
def test_invalid_options_rejected(api, invoice_pdf, options):
    assert submit(api[0], invoice_pdf, **options).status_code == 422


def test_options_must_be_json(api, invoice_pdf):
    r = api[0].post(
        "/jobs", files={"file": ("a.pdf", invoice_pdf, "application/pdf")}, data={"options": "{bad"}
    )
    assert r.status_code == 422


def test_non_pdf_rejected(api):
    assert submit(api[0], b"hello world", "notes.pdf", extract_data=False).status_code == 415


def test_too_large_rejected(base_settings, invoice_pdf):
    settings, engine = base_settings
    small = settings.model_copy(update={"api": settings.api.model_copy(update={"max_upload_mb": 0.001})})
    with TestClient(create_app(small, Factory(), engine)) as client:
        assert submit(client, invoice_pdf, extract_data=False).status_code == 413


def test_unusable_provider_rejected_up_front(api, invoice_pdf):
    client, factory = api

    def broken(llm):
        raise ValueError("no Ollama model configured")

    client.app.state.manager.provider_factory = broken
    r = submit(client, invoice_pdf, instructions="x", provider="ollama")
    assert r.status_code == 400 and "not usable" in r.json()["detail"]


# --- job list / delete / eviction -----------------------------------------------------------


def test_list_and_delete(api, invoice_pdf):
    client, _ = api
    job_id = wait(client, submit(client, invoice_pdf, extract_data=False).json()["id"])["id"]
    assert job_id in [j["id"] for j in client.get("/jobs").json()]
    assert client.delete(f"/jobs/{job_id}").status_code == 204
    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert client.delete(f"/jobs/{job_id}").status_code == 404


def test_oldest_finished_jobs_are_evicted(base_settings):
    settings, engine = base_settings
    two = settings.model_copy(update={"api": settings.api.model_copy(update={"max_jobs": 2})})
    with pymupdf.open() as d:
        add_digital(d)
        pdf = d.tobytes()
    with TestClient(create_app(two, Factory(), engine)) as client:
        ids = []
        for _ in range(3):
            ids.append(submit(client, pdf, extract_data=False).json()["id"])
            wait(client, ids[-1])
        remaining = [j["id"] for j in client.get("/jobs").json()]
        assert remaining == [ids[2], ids[1]]
