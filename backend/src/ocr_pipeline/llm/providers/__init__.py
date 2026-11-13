"""Provider factory. SDK imports stay inside each provider module."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from ocr_pipeline.config import LLMSettings
from ocr_pipeline.llm.providers.base import (
    LLMError,
    LLMProvider,
    LLMRefusal,
    LLMResponse,
    LLMTruncated,
)

# backend/.env holds API keys; real environment variables take precedence.
_ENV_FILE = Path(__file__).parents[4] / ".env"


def create_provider(settings: LLMSettings) -> LLMProvider:
    load_dotenv(_ENV_FILE, override=False)
    if settings.provider == "anthropic":
        from ocr_pipeline.llm.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings.anthropic)
    if settings.provider == "openai":
        from ocr_pipeline.llm.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(settings.openai)
    if settings.provider == "ollama":
        from ocr_pipeline.llm.providers.ollama_provider import OllamaProvider

        return OllamaProvider(settings.ollama)
    raise ValueError(f"unknown LLM provider: {settings.provider}")


__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMRefusal",
    "LLMResponse",
    "LLMTruncated",
    "create_provider",
]
