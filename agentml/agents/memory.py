"""Agent memory: a sliding-window short-term buffer and a persistent, semantically searchable long-term store."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agentml.llm.base import Message


class ShortTermMemory:
    """Keeps the system prompt plus the last ``max_messages`` turns, never splitting a tool exchange."""

    def __init__(self, max_messages: int = 30) -> None:
        self.max_messages = max_messages
        self.messages: list[Message] = []

    def add(self, msg: Message) -> None:
        self.messages.append(msg)
        self._trim()

    def extend(self, msgs: list[Message]) -> None:
        for m in msgs:
            self.add(m)

    def _trim(self) -> None:
        system = [m for m in self.messages if m.role == "system"]
        rest = [m for m in self.messages if m.role != "system"]
        if len(rest) <= self.max_messages:
            return
        rest = rest[-self.max_messages :]
        while rest and rest[0].role == "tool":  # don't start with an orphan tool result
            rest.pop(0)
        self.messages = system + rest

    def clear(self) -> None:
        self.messages = [m for m in self.messages if m.role == "system"]


@dataclass
class MemoryItem:
    text: str
    kind: str = "fact"
    metadata: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)


class LongTermMemory:
    """JSON-backed episodic/semantic memory with embedding-based recall."""

    def __init__(self, path: str | Path | None = None, embedder: Any | None = None) -> None:
        from agentml.rag.embeddings import HashingEmbedder

        self.path = Path(path) if path else None
        self.embedder = embedder or HashingEmbedder()
        self.items: list[MemoryItem] = []
        if self.path and self.path.exists():
            self.items = [MemoryItem(**d) for d in json.loads(self.path.read_text(encoding="utf-8"))]

    def remember(self, text: str, kind: str = "fact", **metadata: Any) -> MemoryItem:
        item = MemoryItem(text=text, kind=kind, metadata=metadata)
        self.items.append(item)
        self.save()
        return item

    def recall(self, query: str, k: int = 3, kind: str | None = None) -> list[tuple[MemoryItem, float]]:
        pool = [m for m in self.items if kind is None or m.kind == kind]
        if not pool:
            return []
        vecs = self.embedder.embed([m.text for m in pool])
        q = self.embedder.embed([query])[0]
        sims = vecs @ q
        order = np.argsort(sims)[::-1][:k]
        return [(pool[i], float(sims[i])) for i in order if sims[i] > 0]

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([asdict(m) for m in self.items], indent=2), encoding="utf-8")

    def __len__(self) -> int:
        return len(self.items)
