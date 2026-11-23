"""Tool registry: turn plain Python functions into LLM-callable tools.

``@tool`` reads the function signature + type hints + docstring and produces a JSON schema,
so the same function works with OpenAI function-calling, Gemini, Ollama and any OpenAI-compatible API.
"""

from __future__ import annotations

import inspect
import json
import time
import traceback
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _json_type(annotation: Any) -> dict[str, Any]:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        non_none = [a for a in args if a is not type(None)]
        return _json_type(non_none[0]) if non_none else {"type": "string"}
    if origin is typing.Literal:
        return {"type": "string", "enum": list(args)}
    if origin in (list, tuple, set):
        item = _json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if origin is dict:
        return {"type": "object"}
    if annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}
    return {"type": "string"}


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return (summary, {param: description}) from a Google-style docstring."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary_lines, params, in_args, current = [], {}, False, None
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in {"args:", "arguments:", "parameters:"}:
            in_args = True
            continue
        if stripped.lower() in {"returns:", "raises:", "example:", "examples:"}:
            in_args = False
            continue
        if in_args:
            if ":" in stripped and not line.startswith(" " * 8):
                name, _, desc = stripped.partition(":")
                current = name.split("(")[0].strip()
                params[current] = desc.strip()
            elif current and stripped:
                params[current] += " " + stripped
        elif stripped:
            summary_lines.append(stripped)
    return " ".join(summary_lines), params


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None):
    """Decorator that attaches a ``Tool`` spec to ``fn`` (as ``fn.tool``)."""

    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        summary, param_docs = _parse_docstring(f.__doc__ or "")
        hints = typing.get_type_hints(f)
        props: dict[str, Any] = {}
        required: list[str] = []
        for pname, p in inspect.signature(f).parameters.items():
            if pname in {"self", "cls"} or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            spec = _json_type(hints.get(pname, str))
            if pname in param_docs:
                spec["description"] = param_docs[pname]
            if p.default is inspect.Parameter.empty:
                required.append(pname)
            elif p.default is not None:
                spec["default"] = p.default
            props[pname] = spec
        f.tool = Tool(  # type: ignore[attr-defined]
            name=name or f.__name__,
            description=description or summary or f.__name__,
            fn=f,
            parameters={"type": "object", "properties": props, "required": required},
        )
        return f

    return wrap(fn) if fn is not None else wrap


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: Any
    seconds: float

    def to_text(self, limit: int = 6000) -> str:
        if isinstance(self.output, str):
            text = self.output
        else:
            text = json.dumps(self.output, default=str, ensure_ascii=False)
        return text if len(text) <= limit else text[:limit] + f"... [truncated {len(text) - limit} chars]"


class ToolRegistry:
    def __init__(self, tools: list[Tool | Callable[..., Any]] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool | Callable[..., Any]) -> Tool:
        spec = t if isinstance(t, Tool) else getattr(t, "tool", None)
        if spec is None:
            spec = tool(t).tool  # type: ignore[union-attr]
        self._tools[spec.name] = spec
        return spec

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        start = time.perf_counter()
        if name not in self._tools:
            return ToolResult(name, False, f"Error: unknown tool '{name}'. Available: {self.names()}", 0.0)
        t = self._tools[name]
        args = dict(arguments or {})
        allowed = set(t.parameters.get("properties", {}))
        unknown = set(args) - allowed
        for k in unknown:
            args.pop(k)
        try:
            out = t.fn(**args)
            return ToolResult(name, True, out, time.perf_counter() - start)
        except Exception as exc:  # tools must never crash the agent loop
            tb = traceback.format_exc(limit=2)
            return ToolResult(name, False, f"Error: {type(exc).__name__}: {exc}\n{tb}", time.perf_counter() - start)
