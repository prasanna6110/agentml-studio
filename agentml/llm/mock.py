"""A deterministic, offline "LLM" so the whole system runs (and is tested) without any API key.

It is not a language model — it is a small rule-based policy that mimics the *protocol* of one:

* When an agent prompt embeds a ``<default_decision>`` JSON block (the analytic heuristic the
  agent computed), the mock returns it verbatim. A real LLM is free to improve on it.
* When a prompt embeds a ``<draft>`` block, the mock returns the draft text.
* When tools are offered, it routes the user request to the best-matching tool by keyword
  overlap and fills arguments from the request — exercising the full tool-calling loop.
* After tool results arrive, it summarises them into a final answer.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agentml.llm.base import LLM, LLMResponse, Message, ToolCall

_DECISION_RE = re.compile(r"<default_decision>(.*?)</default_decision>", re.DOTALL)
_DRAFT_RE = re.compile(r"<draft>(.*?)</draft>", re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9_]+")
_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "in", "on", "for", "with", "me",
    "my", "what", "how", "can", "you", "please", "this", "that", "it", "be", "do", "i", "from",
}


def _words(text: str) -> set[str]:
    out = set()
    for w in _WORD_RE.findall(text.lower()):
        if w in _STOP:
            continue
        for part in {w, *w.split("_")}:
            if part and part not in _STOP:
                out.add(part)
                if len(part) >= 5:  # crude stemming: "calculate" ~ "calculator", "datasets" ~ "dataset"
                    out.add(part[:6] + "~")
    return out


class MockLLM(LLM):
    provider = "mock"

    def __init__(self, model: str = "mock-reasoner-1", temperature: float = 0.0, max_tokens: int = 2048):
        super().__init__(model, temperature, max_tokens)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _route(query: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
        q = _words(query)
        best, best_score = None, 0.0
        for t in tools:
            name_words = _words(t["name"].replace("_", " "))
            desc_words = _words(t.get("description", ""))
            score = 2.0 * len(q & name_words) + 0.5 * len(q & desc_words)
            if score > best_score:
                best, best_score = t, score
        return best if best_score >= 1.0 else None

    @staticmethod
    def _fill_args(tool: dict[str, Any], query: str, context: str) -> dict[str, Any] | None:
        params = tool.get("parameters", {})
        props: dict[str, Any] = params.get("properties", {})
        required = set(params.get("required", []))
        args: dict[str, Any] = {}
        csv = re.search(r"[\w./\\-]+\.csv", query) or re.search(r"[\w./\\-]+\.csv", context)
        for name, spec in props.items():
            lname = name.lower()
            value: Any = None
            if lname in {"query", "question", "text", "prompt", "topic", "goal", "request"}:
                value = query
            elif lname in {"expression", "expr", "formula"}:
                m = re.search(r"[-+*/().\d\s^%]{3,}", query)
                value = m.group(0).strip().replace("^", "**") if m else None
            elif lname in {"path", "dataset", "file", "csv_path", "dataset_path"}:
                value = csv.group(0) if csv else None
            elif lname in {"target", "target_column", "label"}:
                m = re.search(r"(?:target|label|predict)\s*(?:column)?\s*[:=]?\s*['\"]?([A-Za-z_][\w]*)", query)
                value = m.group(1) if m and m.group(1).lower() not in _STOP else None
            elif spec.get("type") == "integer" and "default" in spec:
                value = spec["default"]
            if value is not None:
                args[name] = value
        if not required.issubset(args):
            return None
        return args

    @staticmethod
    def _render_value(v: Any, depth: int = 0) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, dict):
            if depth >= 1 or len(v) > 8:
                items = list(v.items())[:6]
                return ", ".join(f"{k}: {MockLLM._render_value(x, 2)}" for k, x in items) + (" …" if len(v) > 6 else "")
            return "\n" + "\n".join(f"  - **{k}**: {MockLLM._render_value(x, depth + 1)}" for k, x in v.items())
        if isinstance(v, list):
            if not v:
                return "none"
            if all(isinstance(x, (str, int, float)) for x in v):
                return ", ".join(MockLLM._render_value(x, 2) for x in v[:12]) + (" …" if len(v) > 12 else "")
            return f"{len(v)} items, e.g. " + MockLLM._render_value(v[0], 2)
        text = str(v)
        return text if len(text) <= 300 else text[:300] + " …"

    @staticmethod
    def _summarise_tool_results(messages: list[Message]) -> str:
        results = []
        for m in reversed(messages):
            if m.role != "tool":
                break
            results.append((m.name, m.content))
        results.reverse()
        parts = []
        for name, content in results:
            title = (name or "tool").replace("_", " ")
            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                parsed = content
            if isinstance(parsed, dict) and "answer" in parsed:
                parts.append(str(parsed["answer"]))
            elif isinstance(parsed, dict):
                skip = {"columns"} if "columns" in parsed and len(parsed) > 3 else set()
                lines = [f"- **{k.replace('_', ' ')}**: {MockLLM._render_value(v)}" for k, v in parsed.items() if k not in skip]
                parts.append(f"**Result of {title}**\n\n" + "\n".join(lines))
            elif isinstance(parsed, list):
                parts.append(f"**Result of {title}**\n\n" + "\n".join(f"- {MockLLM._render_value(x, 1)}" for x in parsed[:20]))
            else:
                parts.append(f"**Result of {title}**: {MockLLM._render_value(parsed)}")
        parts.append("_Offline mode: add an API key in Settings and the copilot will explain these results in its own words._")
        return "\n\n".join(parts)

    # ---------------------------------------------------------------- protocol
    def _chat(self, messages, tools, temperature, max_tokens) -> LLMResponse:
        last = messages[-1] if messages else Message.user("")
        full_prompt = "\n".join(m.content for m in messages if m.content)
        usage = {"input_tokens": len(full_prompt) // 4, "output_tokens": 0}

        def done(text: str, calls: list[ToolCall] | None = None) -> LLMResponse:
            usage["output_tokens"] = len(text) // 4 + 8 * len(calls or [])
            return LLMResponse(text, calls or [], usage, "tool_use" if calls else "end_turn")

        if last.role == "tool":
            return done(self._summarise_tool_results(messages))

        if m := _DECISION_RE.search(last.content):
            return done(m.group(1).strip())
        if m := _DRAFT_RE.search(last.content):
            return done(m.group(1).strip())

        if tools and last.role == "user":
            tool = self._route(last.content, tools)
            if tool is not None:
                context = "\n".join(m.content for m in messages if m.role == "system")
                args = self._fill_args(tool, last.content, context)
                if args is not None:
                    return done(f"I'll use `{tool['name']}` for this.", [ToolCall(tool["name"], args)])

        return done(
            "(mock mode) I'm running without a real language model. Add an OpenAI or Gemini "
            "API key in Settings for full reasoning. "
            f"Your request was: {last.content[:300]}"
        )
