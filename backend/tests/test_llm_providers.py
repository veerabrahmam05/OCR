"""Provider adapters against fake SDK clients: request shape and response handling."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest

from ocr_pipeline.config import (
    AnthropicSettings,
    LLMSettings,
    OllamaSettings,
    OpenAISettings,
)
from ocr_pipeline.llm.providers import create_provider
from ocr_pipeline.llm.providers.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from ocr_pipeline.llm.providers.base import LLMRefusal, LLMTruncated
from ocr_pipeline.llm.providers.ollama_provider import OllamaProvider
from ocr_pipeline.llm.providers.openai_provider import OpenAIProvider

SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


# --- Anthropic ------------------------------------------------------------------------


class FakeAnthropic:
    def __init__(self, message):
        self.message = message
        self.kwargs = None
        self.beta = NS(messages=NS(stream=self._stream))

    @contextmanager
    def _stream(self, **kwargs):
        self.kwargs = kwargs
        yield NS(get_final_message=lambda: self.message)


def claude_message(stop_reason="end_turn", text='{"data": {}, "sources": []}', details=None):
    content = [NS(type="thinking", thinking=""), NS(type="text", text=text)] if text else []
    return NS(
        stop_reason=stop_reason,
        stop_details=details,
        content=content,
        model="claude-opus-5-5",
        usage=NS(input_tokens=1200, output_tokens=80),
    )


def test_anthropic_request_shape():
    client = FakeAnthropic(claude_message())
    provider = AnthropicProvider(AnthropicSettings(effort="high"), client=client)
    response = provider.complete_json("sys", "user text", SCHEMA)

    kw = client.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["system"] == "sys"
    assert kw["messages"] == [{"role": "user", "content": "user text"}]
    assert kw["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert kw["max_tokens"] == 64_000
    assert "thinking" not in kw  # always on for Opus 5.5; effort is the control
    assert (response.text, response.input_tokens, response.output_tokens) == (
        '{"data": {}, "sources": []}', 1200, 80,
    )  # fmt: skip


def test_anthropic_refusal_and_truncation():
    refused = claude_message("refusal", text=None, details=NS(category="cyber"))
    with pytest.raises(LLMRefusal, match="cyber"):
        AnthropicProvider(AnthropicSettings(), client=FakeAnthropic(refused)).complete_json("s", "u", SCHEMA)
    cut = claude_message("max_tokens")
    with pytest.raises(LLMTruncated):
        AnthropicProvider(AnthropicSettings(), client=FakeAnthropic(cut)).complete_json("s", "u", SCHEMA)


# --- OpenAI ---------------------------------------------------------------------------


class FakeOpenAI:
    def __init__(self, content='{"data": {}, "sources": []}', refusal=None, finish="stop"):
        self.kwargs = None
        self.reply = NS(
            choices=[NS(message=NS(content=content, refusal=refusal), finish_reason=finish)],
            model="gpt-5",
            usage=NS(prompt_tokens=900, completion_tokens=60),
        )
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self.reply


def test_openai_request_shape():
    client = FakeOpenAI()
    response = OpenAIProvider(OpenAISettings(), client=client).complete_json("sys", "user", SCHEMA)
    kw = client.kwargs
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
    assert kw["response_format"]["json_schema"] == {"name": "extraction", "schema": SCHEMA, "strict": True}
    assert kw["max_completion_tokens"] == 16_000
    assert response.input_tokens == 900


def test_openai_refusal_and_truncation():
    with pytest.raises(LLMRefusal, match="no"):
        OpenAIProvider(OpenAISettings(), client=FakeOpenAI(content=None, refusal="no")).complete_json(
            "s", "u", SCHEMA
        )
    with pytest.raises(LLMTruncated):
        OpenAIProvider(OpenAISettings(), client=FakeOpenAI(finish="length")).complete_json("s", "u", SCHEMA)


# --- Ollama ---------------------------------------------------------------------------


class FakeOllama:
    def __init__(self, done_reason="stop"):
        self.kwargs = None
        self.done_reason = done_reason

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return NS(
            message=NS(content='{"data": {}, "sources": []}'),
            done_reason=self.done_reason,
            model="qwen2.5:7b",
            prompt_eval_count=700,
            eval_count=40,
        )


def test_ollama_request_shape():
    client = FakeOllama()
    settings = OllamaSettings(model="qwen2.5:7b")
    response = OllamaProvider(settings, client=client).complete_json("sys", "user", SCHEMA)
    kw = client.kwargs
    assert kw["format"] == SCHEMA
    assert kw["options"] == {"num_ctx": 16_384, "num_predict": 4_096, "temperature": 0, "presence_penalty": 0}
    assert kw["think"] is False  # off by default
    assert response.output_tokens == 40


def test_ollama_think_left_to_model_when_unset():
    client = FakeOllama()
    OllamaProvider(OllamaSettings(model="m", think=None), client=client).complete_json("s", "u", SCHEMA)
    assert "think" not in client.kwargs


def test_ollama_truncation():
    with pytest.raises(LLMTruncated):
        OllamaProvider(OllamaSettings(model="m"), client=FakeOllama("length")).complete_json("s", "u", SCHEMA)


def test_ollama_requires_a_model():
    with pytest.raises(ValueError, match="no Ollama model"):
        OllamaProvider(OllamaSettings(), client=FakeOllama())


def test_ollama_budget_must_fit_context():
    with pytest.raises(ValueError, match="num_ctx"):
        OllamaSettings(model="m", num_ctx=8_192, max_input_tokens=8_000, max_output_tokens=4_096)


# --- factory --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider, cls", [("anthropic", AnthropicProvider), ("openai", OpenAIProvider)]
)
def test_factory_builds_selected_provider(monkeypatch, provider, cls):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert isinstance(create_provider(LLMSettings(provider=provider)), cls)


def test_factory_ollama_without_model_fails_clearly():
    with pytest.raises(ValueError, match="llm.ollama"):
        create_provider(LLMSettings(provider="ollama"))
