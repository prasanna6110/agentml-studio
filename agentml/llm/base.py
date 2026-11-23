"""Provider-agnostic message / tool-call types and the abstract ``LLM`` interface."""

from __future__ import annotations

import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name for role == "tool"

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls("assistant", content, tool_calls or [])

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str) -> Message:
        return cls("tool", content, tool_call_id=tool_call_id, name=name)


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    """Raised when a provider call fails after retries."""


class LLM(ABC):
    """Minimal chat interface every provider implements.

    ``tools`` use a neutral schema: ``{"name", "description", "parameters": <JSON schema>}``.
    Each provider converts to / from its own wire format.
    """

    provider: str = "base"

    def __init__(self, model: str, temperature: float = 0.2, max_tokens: int = 2048) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    @abstractmethod
    def _chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse: ...

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        resp = self._chat(
            messages,
            tools or None,
            self.temperature if temperature is None else temperature,
            max_tokens or self.max_tokens,
        )
        self.total_usage["calls"] += 1
        self.total_usage["input_tokens"] += int(resp.usage.get("input_tokens", 0))
        self.total_usage["output_tokens"] += int(resp.usage.get("output_tokens", 0))
        return resp

    def complete(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        msgs = [Message.system(system)] if system else []
        msgs.append(Message.user(prompt))
        return self.chat(msgs, **kw).content

    def complete_json(
        self, prompt: str, system: str | None = None, default: dict | None = None, **kw: Any
    ) -> dict[str, Any]:
        """Ask for a JSON object and parse it robustly; return ``default`` on failure."""
        text = self.complete(prompt, system=system, **kw)
        parsed = extract_json(text)
        if isinstance(parsed, dict):
            return parsed
        return dict(default or {})

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of free-form LLM text."""
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            pass
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = cand.find(open_ch)
            if start == -1:
                continue
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cand[start : i + 1])
                        except (json.JSONDecodeError, ValueError):
                            break
    return None
