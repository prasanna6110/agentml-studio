"""HTTP clients for OpenAI (and any OpenAI-compatible API), Google Gemini and Ollama.

All providers speak raw HTTPS through ``httpx`` — no vendor SDKs required — and support
native tool / function calling, retries with exponential backoff, and token accounting.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from agentml.llm.base import LLM, LLMError, LLMResponse, Message, ToolCall

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class _HTTPProvider(LLM):
    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 90.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(model, temperature, max_tokens)
        self.max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout)

    def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:  # network error -> retry
                last_err = exc
            else:
                if r.status_code < 400:
                    return r.json()
                if r.status_code not in RETRY_STATUS:
                    raise LLMError(f"{self.provider} HTTP {r.status_code}: {r.text[:500]}")
                last_err = LLMError(f"{self.provider} HTTP {r.status_code}: {r.text[:200]}")
                retry_after = r.headers.get("retry-after")
                if retry_after and retry_after.replace(".", "", 1).isdigit():
                    time.sleep(min(float(retry_after), 30))
                    continue
            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 20) * 0.75)
        raise LLMError(f"{self.provider} request failed after retries: {last_err}")


# --------------------------------------------------------------------------- OpenAI
class OpenAILLM(_HTTPProvider):
    """Works with OpenAI and any OpenAI-compatible endpoint (Groq, Together, vLLM, LM Studio...)."""

    provider = "openai"

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1", **kw: Any):
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        super().__init__(model, **kw)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def to_wire(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.role == "assistant" and m.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": m.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    _use_completion_tokens = False
    _omit_temperature = False

    def _post_adaptive(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        """Newer OpenAI models reject `max_tokens` / custom `temperature`; adapt once and remember."""
        for _ in range(3):
            body = dict(payload)
            if self._use_completion_tokens and "max_tokens" in body:
                body["max_completion_tokens"] = body.pop("max_tokens")
            if self._omit_temperature:
                body.pop("temperature", None)
            try:
                return self._post(f"{self.base_url}/chat/completions", body, headers)
            except LLMError as exc:
                msg = str(exc)
                if "max_tokens" in msg and not self._use_completion_tokens:
                    self._use_completion_tokens = True
                elif "temperature" in msg and not self._omit_temperature:
                    self._omit_temperature = True
                else:
                    raise
        return self._post(f"{self.base_url}/chat/completions", body, headers)

    def _chat(self, messages, tools, temperature, max_tokens) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.to_wire(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = self._post_adaptive(payload, headers)
        choice = data["choices"][0]
        msg = choice.get("message", {})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments")}
            calls.append(ToolCall(fn.get("name", ""), args, tc.get("id") or ToolCall("x").id))
        usage = data.get("usage", {})
        return LLMResponse(
            msg.get("content") or "",
            calls,
            {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            choice.get("finish_reason"),
            data,
        )


# --------------------------------------------------------------------------- Gemini
_GEMINI_DROP_KEYS = {"default", "additionalProperties", "$schema", "title", "examples"}


def _gemini_schema(schema: Any) -> Any:
    """Gemini accepts an OpenAPI subset of JSON schema — strip unsupported keys recursively."""
    if isinstance(schema, dict):
        return {k: _gemini_schema(v) for k, v in schema.items() if k not in _GEMINI_DROP_KEYS}
    if isinstance(schema, list):
        return [_gemini_schema(v) for v in schema]
    return schema


class GeminiLLM(_HTTPProvider):
    provider = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str, **kw: Any) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        super().__init__(model, **kw)
        self.api_key = api_key

    @staticmethod
    def to_wire(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                part = {"functionResponse": {"name": m.name, "response": {"content": m.content}}}
                if contents and contents[-1]["role"] == "user" and "functionResponse" in contents[-1]["parts"][0]:
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
                continue
            role = "model" if m.role == "assistant" else "user"
            parts: list[dict[str, Any]] = []
            if m.content:
                parts.append({"text": m.content})
            for tc in m.tool_calls:
                parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
            contents.append({"role": role, "parts": parts or [{"text": "(empty)"}]})
        return system, contents

    def _chat(self, messages, tools, temperature, max_tokens) -> LLMResponse:
        system, contents = self.to_wire(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t["name"],
                            "description": t["description"],
                            "parameters": _gemini_schema(t["parameters"]),
                        }
                        for t in tools
                    ]
                }
            ]
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        data = self._post(f"{self.BASE}/{self.model}:generateContent", payload, headers)
        cands = data.get("candidates") or [{}]
        parts = (cands[0].get("content") or {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        calls = [
            ToolCall(p["functionCall"]["name"], p["functionCall"].get("args") or {})
            for p in parts
            if "functionCall" in p
        ]
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text,
            calls,
            {
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
            },
            cands[0].get("finishReason"),
            data,
        )


# --------------------------------------------------------------------------- Ollama
class OllamaLLM(_HTTPProvider):
    """Local open-weight models (Llama, Qwen, Mistral...) via an Ollama server."""

    provider = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434", **kw: Any) -> None:
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")

    def _chat(self, messages, tools, temperature, max_tokens) -> LLMResponse:
        wire = []
        for m in messages:
            item: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                item["tool_calls"] = [
                    {"function": {"name": tc.name, "arguments": tc.arguments}} for tc in m.tool_calls
                ]
            wire.append(item)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": wire,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        data = self._post(f"{self.base_url}/api/chat", payload, {"Content-Type": "application/json"})
        msg = data.get("message", {})
        calls = [
            ToolCall(tc["function"]["name"], tc["function"].get("arguments") or {})
            for tc in msg.get("tool_calls") or []
        ]
        return LLMResponse(
            msg.get("content", ""),
            calls,
            {"input_tokens": data.get("prompt_eval_count", 0), "output_tokens": data.get("eval_count", 0)},
            data.get("done_reason"),
            data,
        )
