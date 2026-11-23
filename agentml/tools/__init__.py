"""Tool system: decorate Python functions to make them callable by any LLM provider."""

from agentml.tools.builtin import build_copilot_tools, calculator
from agentml.tools.registry import Tool, ToolRegistry, ToolResult, tool

__all__ = ["Tool", "ToolRegistry", "ToolResult", "build_copilot_tools", "calculator", "tool"]
