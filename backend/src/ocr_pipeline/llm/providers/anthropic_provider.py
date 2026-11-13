"""Claude via the official Anthropic SDK, using JSON structured outputs."""

from __future__ import annotations

import anthropic

from ocr_pipeline.config import AnthropicSettings
from ocr_pipeline.llm.providers.base import (
    LLMProvider,
    LLMRefusal,
    LLMResponse,
    LLMTruncated,
)

# On a safety-classifier decline the API re-runs the request on a fallback model
# chosen by refusal category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: AnthropicSettings, client: anthropic.Anthropic | None = None):
        self.model = settings.model
        self.effort = settings.effort
        self.max_input_tokens = settings.max_input_tokens
        self.max_output_tokens = settings.max_output_tokens
        # Credentials resolve from ANTHROPIC_API_KEY (or an `ant auth login` profile).
        self.client = client or anthropic.Anthropic()

    def complete_json(self, system: str, user: str, schema: dict) -> LLMResponse:
        # Streamed: documents can be long and max_tokens is high.
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            message = stream.get_final_message()

        if message.stop_reason == "refusal":
            details = message.stop_details
            category = getattr(details, "category", None) if details else None
            raise LLMRefusal(f"Claude declined the request (category: {category})")
        if message.stop_reason == "max_tokens":
            raise LLMTruncated(
                f"output hit max_tokens={self.max_output_tokens}; raise [llm.anthropic].max_output_tokens"
            )

        text = next((b.text for b in message.content if b.type == "text"), None)
        if text is None:
            raise LLMRefusal(f"no text in response (stop_reason={message.stop_reason})")
        return LLMResponse(
            text=text,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
