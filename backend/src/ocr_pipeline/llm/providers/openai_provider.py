"""OpenAI chat completions with strict JSON-schema response format."""

from __future__ import annotations

import openai

from ocr_pipeline.config import OpenAISettings
from ocr_pipeline.llm.providers.base import (
    LLMProvider,
    LLMRefusal,
    LLMResponse,
    LLMTruncated,
)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, settings: OpenAISettings, client: openai.OpenAI | None = None):
        self.model = settings.model
        self.max_input_tokens = settings.max_input_tokens
        self.max_output_tokens = settings.max_output_tokens
        # Credentials resolve from OPENAI_API_KEY.
        self.client = client or openai.OpenAI(base_url=settings.base_url)

    def complete_json(self, system: str, user: str, schema: dict) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=self.max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": schema, "strict": True},
            },
        )
        choice = response.choices[0]
        if choice.message.refusal:
            raise LLMRefusal(f"OpenAI declined the request: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise LLMTruncated(
                f"output hit max_completion_tokens={self.max_output_tokens}; "
                "raise [llm.openai].max_output_tokens"
            )
        if not choice.message.content:
            raise LLMRefusal(f"empty response (finish_reason={choice.finish_reason})")
        usage = response.usage
        return LLMResponse(
            text=choice.message.content,
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
