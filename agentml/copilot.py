"""The Data-Science Copilot: a conversational ReAct agent with ML, RAG and math tools."""

from __future__ import annotations

from agentml.agents.base import Agent
from agentml.agents.memory import LongTermMemory
from agentml.config import Settings, get_settings
from agentml.llm.base import LLM
from agentml.rag.agentic_rag import AgenticRAG
from agentml.tools.builtin import build_copilot_tools


def build_copilot(llm: LLM, rag: AgenticRAG | None = None, settings: Settings | None = None) -> Agent:
    s = settings or get_settings()
    s.ensure_workspace()
    tools = build_copilot_tools(s.workspace / "datasets", rag=rag, runs_dir=s.workspace / "runs", llm=llm)
    datasets = ", ".join(tools.execute("list_datasets").output or []) or "none uploaded yet"
    return Agent(
        name="Copilot",
        role="an expert AI data-science copilot",
        llm=llm,
        tools=tools,
        max_steps=6,
        long_term=LongTermMemory(s.workspace / "memory" / "copilot_memory.json"),
        instructions=(
            "Help the user explore data, train models and answer questions from their documents. "
            "Use profile_dataset before training on an unfamiliar dataset, train_model to build models, "
            "search_knowledge_base for questions about ingested documents and calculator for arithmetic. "
            f"Available datasets: {datasets}."
        ),
    )
