"""LLM layer: one interface, four backends (OpenAI-compatible, Gemini, Ollama, Mock)."""

from __future__ import annotations

from agentml.config import DEFAULT_MODELS, Settings, get_settings
from agentml.llm.base import LLM, LLMError, LLMResponse, Message, ToolCall, extract_json
from agentml.llm.mock import MockLLM
from agentml.llm.providers import GeminiLLM, OllamaLLM, OpenAILLM


def get_llm(provider: str | None = None, model: str | None = None, settings: Settings | None = None) -> LLM:
    """Build an LLM client from settings (env / .env) with optional overrides."""
    s = settings or get_settings()
    provider = (provider or s.provider).lower()
    model = model or (s.model if provider == s.provider else DEFAULT_MODELS.get(provider, ""))
    common = {
        "temperature": s.temperature,
        "max_tokens": s.max_tokens,
        "timeout": s.request_timeout,
        "max_retries": s.max_retries,
    }
    if provider == "mock":
        return MockLLM(model=model or DEFAULT_MODELS["mock"])
    if provider in {"openai", "gemini", "ollama"} and not model:
        raise LLMError(f"No model name set for {provider}. Enter one in Settings (or AGENTML_MODEL in .env).")
    if provider == "openai":
        return OpenAILLM(s.openai_api_key or "", model, base_url=s.openai_base_url, **common)
    if provider == "gemini":
        return GeminiLLM(s.gemini_api_key or "", model, **common)
    if provider == "ollama":
        return OllamaLLM(model, base_url=s.ollama_base_url, **common)
    raise LLMError(f"Unknown provider {provider!r}")


__all__ = [
    "LLM",
    "GeminiLLM",
    "LLMError",
    "LLMResponse",
    "Message",
    "MockLLM",
    "OllamaLLM",
    "OpenAILLM",
    "ToolCall",
    "extract_json",
    "get_llm",
]
