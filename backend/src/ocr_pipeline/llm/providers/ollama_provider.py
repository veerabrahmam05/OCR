"""Local models through Ollama, with JSON-schema constrained output."""

from __future__ import annotations

import ollama

from ocr_pipeline.config import OllamaSettings
from ocr_pipeline.llm.providers.base import LLMProvider, LLMResponse, LLMTruncated


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, settings: OllamaSettings, client: ollama.Client | None = None):
        if not settings.model:
            raise ValueError('no Ollama model configured: set [llm.ollama].model, e.g. "qwen2.5:7b"')
        self.model = settings.model
        self.num_ctx = settings.num_ctx
        self.think = settings.think
        self.max_input_tokens = settings.max_input_tokens
        self.max_output_tokens = settings.max_output_tokens
        self.client = client or ollama.Client(host=settings.base_url)

    def complete_json(self, system: str, user: str, schema: dict) -> LLMResponse:
        extra = {} if self.think is None else {"think": self.think}
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema,
            options={
                "num_ctx": self.num_ctx,
                "num_predict": self.max_output_tokens,
                "temperature": 0,  # extraction, not creativity
                # Some models ship a presence penalty (qwen3.5: 1.5); extraction must be
                # free to repeat values, e.g. the same amount in a line and the total.
                "presence_penalty": 0,
            },
            **extra,
        )
        if response.done_reason == "length":
            raise LLMTruncated(
                f"output hit num_predict={self.max_output_tokens}; raise [llm.ollama].max_output_tokens"
            )
        return LLMResponse(
            text=response.message.content or "",
            model=response.model or self.model,
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
        )
