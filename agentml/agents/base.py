"""The core agent: an LLM + tools + memory running a ReAct-style tool-calling loop."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentml.agents.memory import LongTermMemory, ShortTermMemory
from agentml.llm.base import LLM, Message, extract_json
from agentml.tools.registry import ToolRegistry

EventHandler = Callable[[dict[str, Any]], None]


@dataclass
class Step:
    kind: str  # "thought" | "tool_call" | "tool_result" | "answer" | "decision"
    content: Any
    tool: str | None = None
    seconds: float = 0.0


@dataclass
class AgentResult:
    agent: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    tool_calls: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "answer": self.answer,
            "tool_calls": self.tool_calls,
            "seconds": round(self.seconds, 2),
            "steps": [
                {"kind": s.kind, "tool": s.tool, "content": s.content, "seconds": round(s.seconds, 3)}
                for s in self.steps
            ],
        }


class Agent:
    """A single autonomous agent.

    Args:
        name: Display name, e.g. "Data Analyst".
        role: One-line job description injected into the system prompt.
        llm: Any ``LLM`` implementation.
        tools: Tools the agent may call.
        instructions: Extra system-prompt guidance.
        max_steps: Upper bound on think→act iterations (guards against loops).
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm: LLM,
        tools: ToolRegistry | None = None,
        instructions: str = "",
        max_steps: int = 6,
        long_term: LongTermMemory | None = None,
        on_event: EventHandler | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.instructions = instructions
        self.max_steps = max_steps
        self.long_term = long_term
        self.on_event = on_event
        self.memory = ShortTermMemory()
        self.memory.add(Message.system(self.system_prompt()))

    def system_prompt(self) -> str:
        parts = [
            f"You are {self.name}, {self.role}.",
            "You are part of AgentML Studio, a team of AI agents that build machine-learning solutions.",
            "Think step by step, be precise and quantitative, and never invent numbers you did not observe.",
        ]
        if len(self.tools):
            parts.append(
                "You can call tools. Call a tool whenever it gives you facts you need; "
                "when you have enough information, answer directly without calling tools."
            )
        if self.instructions:
            parts.append(self.instructions)
        return "\n".join(parts)

    def emit(self, kind: str, **data: Any) -> None:
        if self.on_event:
            self.on_event({"agent": self.name, "kind": kind, "time": time.time(), **data})

    # ------------------------------------------------------------------ ReAct loop
    def run(self, task: str, context: str | None = None) -> AgentResult:
        start = time.perf_counter()
        result = AgentResult(agent=self.name, answer="")
        content = task
        if self.long_term is not None and len(self.long_term):
            recalled = self.long_term.recall(task, k=3)
            if recalled:
                content += "\n\nRelevant memories:\n" + "\n".join(f"- {m.text}" for m, _ in recalled)
        if context:
            content += f"\n\nContext:\n{context}"
        self.memory.add(Message.user(content))
        self.emit("start", task=task)

        for _ in range(self.max_steps):
            resp = self.llm.chat(self.memory.messages, tools=self.tools.schemas() if len(self.tools) else None)
            if not resp.tool_calls:
                self.memory.add(Message.assistant(resp.content))
                result.answer = resp.content
                result.steps.append(Step("answer", resp.content))
                self.emit("answer", content=resp.content)
                break
            if resp.content:
                result.steps.append(Step("thought", resp.content))
                self.emit("thought", content=resp.content)
            self.memory.add(Message.assistant(resp.content, resp.tool_calls))
            for call in resp.tool_calls:
                self.emit("tool_call", tool=call.name, arguments=call.arguments)
                result.steps.append(Step("tool_call", call.arguments, tool=call.name))
                out = self.tools.execute(call.name, call.arguments)
                result.tool_calls += 1
                text = out.to_text()
                result.steps.append(Step("tool_result", text[:2000], tool=call.name, seconds=out.seconds))
                self.emit("tool_result", tool=call.name, ok=out.ok, preview=text[:400])
                self.memory.add(Message.tool(text, call.id, call.name))
        else:
            # step budget exhausted: force a final answer without tools
            resp = self.llm.chat(
                self.memory.messages + [Message.user("Step limit reached. Give your best final answer now.")]
            )
            result.answer = resp.content
            result.steps.append(Step("answer", resp.content))
            self.memory.add(Message.assistant(resp.content))

        result.seconds = time.perf_counter() - start
        return result

    # ------------------------------------------------------------------ structured decisions
    def decide(self, question: str, default: dict[str, Any], evidence: dict[str, Any] | str,
               allowed_keys: set[str] | None = None) -> dict[str, Any]:
        """Ask the agent for a JSON decision, seeded with an analytic default.

        The LLM sees the evidence and a default decision computed by deterministic analysis.
        It may keep or improve it; the output is parsed, validated against ``allowed_keys`` and
        merged over the default, so a malformed reply can never break the pipeline.
        """
        ev = evidence if isinstance(evidence, str) else json.dumps(evidence, default=str, indent=1)[:12000]
        prompt = (
            f"{question}\n\nEvidence:\n{ev}\n\n"
            "A baseline decision computed by deterministic analysis is below. Keep what is right, "
            "improve what is not, and reply with ONLY a JSON object with the same keys plus a short "
            '"rationale" string.\n'
            f"<default_decision>\n{json.dumps(default, default=str)}\n</default_decision>"
        )
        start = time.perf_counter()
        reply = self.llm.chat([Message.system(self.system_prompt()), Message.user(prompt)])
        parsed = extract_json(reply.content)
        decision = dict(default)
        if isinstance(parsed, dict):
            keys = allowed_keys or set(default) | {"rationale"}
            decision.update({k: v for k, v in parsed.items() if k in keys})
        self.emit("decision", decision=decision, seconds=round(time.perf_counter() - start, 2))
        return decision

    def write(self, instruction: str, draft: str, evidence: str = "") -> str:
        """Ask the agent to write/improve prose, given a template draft (used by the Reporter)."""
        prompt = f"{instruction}\n\n{evidence}\n\nStart from this draft and improve it:\n<draft>\n{draft}\n</draft>"
        start = time.perf_counter()
        text = self.llm.chat([Message.system(self.system_prompt()), Message.user(prompt)]).content.strip()
        self.emit("log", message=f"Wrote {len(text.split())} words", seconds=round(time.perf_counter() - start, 2))
        return text or draft
