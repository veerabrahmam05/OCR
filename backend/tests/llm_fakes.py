"""Fake provider and document builders for LLM tests (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ocr_pipeline.llm.providers.base import LLMProvider, LLMResponse
from ocr_pipeline.models import (
    ClassificationReason,
    ExtractedDocument,
    ExtractionMethod,
    PageClassification,
    PageContent,
    PageSignals,
    PageType,
)


def make_document(pages_markdown: list[str]) -> ExtractedDocument:
    pages = []
    for i, md in enumerate(pages_markdown, start=1):
        signals = PageSignals(
            char_count=len(md), ocr_char_count=0, word_count=0, bad_char_ratio=0.0,
            image_coverage=0.0, curve_vector_count=0, has_ocr_layer=False,
        )  # fmt: skip
        cls = PageClassification(
            page_number=i, page_type=PageType.DIGITAL, needs_ocr=False,
            reason=ClassificationReason.TEXT_LAYER_OK, threshold=0.6, signals=signals,
        )  # fmt: skip
        pages.append(
            PageContent(
                page_number=i, width=612, height=792, method=ExtractionMethod.TEXT_LAYER,
                markdown=md, classification=cls,
            )  # fmt: skip
        )
    return ExtractedDocument(source="<test>", page_count=len(pages), pages=pages)


Reply = dict | str | Exception | Callable[[str], dict | str]


class FakeProvider(LLMProvider):
    """Returns queued replies in order; records every call."""

    name = "fake"

    def __init__(self, replies: list[Reply], max_input_tokens: int = 100_000):
        self.model = "fake-model"
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = 4_000
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, system: str, user: str, schema: dict) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "schema": schema})
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(user)
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(text=text, model="fake-model-v1", input_tokens=100, output_tokens=20)
