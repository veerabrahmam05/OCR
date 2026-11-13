"""Provider interface: one call, a JSON schema in, schema-conforming JSON text out."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(RuntimeError):
    """Base class for provider failures the extractor should report, not retry blindly."""


class LLMRefusal(LLMError):
    pass


class LLMTruncated(LLMError):
    """Output hit the token cap; the JSON is incomplete."""


@dataclass(frozen=True)
class LLMResponse:
    text: str  # JSON text
    model: str  # the model that actually answered (may differ after a fallback)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(ABC):
    name: str
    model: str
    max_input_tokens: int
    max_output_tokens: int

    @abstractmethod
    def complete_json(self, system: str, user: str, schema: dict) -> LLMResponse:
        """Return JSON text that conforms to ``schema`` (a strict schema from llm.schema)."""
