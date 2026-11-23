"""In-memory vector store with hybrid (dense + BM25) retrieval and reciprocal-rank fusion."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from agentml.rag.chunking import Chunk
from agentml.rag.embeddings import Embedder, HashingEmbedder

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to and or is are in on for with by as at be this that it from was were which can".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[Counter] = []
        self.lengths: list[int] = []
        self.df: Counter = Counter()

    def add(self, texts: list[str]) -> None:
        for t in texts:
            toks = tokenize(t)
            c = Counter(toks)
            self.docs.append(c)
            self.lengths.append(len(toks))
            self.df.update(c.keys())

    def scores(self, query: str) -> np.ndarray:
        n = len(self.docs)
        if n == 0:
            return np.zeros(0)
        avg = sum(self.lengths) / n or 1.0
        out = np.zeros(n)
        for term in set(tokenize(query)):
            df = self.df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self.docs):
                tf = doc.get(term, 0)
                if tf:
                    out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / avg))
        return out


class VectorStore:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self.bm25 = BM25()
        self._ids: set[str] = set()

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk]) -> int:
        new = [c for c in chunks if c.id not in self._ids]
        if not new:
            return 0
        vecs = self.embedder.embed([c.text for c in new])
        self.vectors = np.vstack([self.vectors, vecs]) if len(self.vectors) else vecs
        self.chunks.extend(new)
        self.bm25.add([c.text for c in new])
        self._ids.update(c.id for c in new)
        return len(new)

    def sources(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.source] = out.get(c.source, 0) + 1
        return out

    def remove_source(self, source: str) -> int:
        """Delete every chunk of one document and rebuild the keyword index."""
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else np.zeros((0, self.embedder.dim), dtype=np.float32)
            self.bm25 = BM25()
            self.bm25.add([c.text for c in self.chunks])
            self._ids = {c.id for c in self.chunks}
        return removed

    def dense_search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        sims = self.vectors @ self.embedder.embed([query])[0]
        order = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in order]

    def keyword_search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        s = self.bm25.scores(query)
        if not len(s):
            return []
        order = np.argsort(s)[::-1][:k]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]

    def hybrid_search(self, query: str, k: int = 5, rrf_k: int = 60, dense_weight: float = 1.0,
                      keyword_weight: float = 1.0, sources: set[str] | None = None) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion of dense and BM25 rankings (optionally restricted to some documents)."""
        pool = max(k * 4, 20) if not sources else len(self.chunks)
        dense = self.dense_search(query, pool)
        kw = self.keyword_search(query, pool)
        if sources:
            dense = [(i, s) for i, s in dense if self.chunks[i].source in sources]
            kw = [(i, s) for i, s in kw if self.chunks[i].source in sources]
        fused: dict[int, float] = {}
        for rank, (i, _) in enumerate(dense):
            fused[i] = fused.get(i, 0) + dense_weight / (rrf_k + rank + 1)
        for rank, (i, _) in enumerate(kw):
            fused[i] = fused.get(i, 0) + keyword_weight / (rrf_k + rank + 1)
        dense_map, kw_map = dict(dense), dict(kw)
        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            {
                "chunk": self.chunks[i],
                "score": round(s, 5),
                "dense": round(dense_map.get(i, 0.0), 4),
                "bm25": round(kw_map.get(i, 0.0), 4),
            }
            for i, s in top
        ]

    # ---------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks]), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> VectorStore:
        d = Path(directory)
        store = cls(embedder)
        if not (d / "chunks.json").exists():
            return store
        chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text(encoding="utf-8"))]
        vectors = np.load(d / "vectors.npy")
        if vectors.shape[1] != store.embedder.dim:  # embedder changed -> re-embed
            store.add(chunks)
            return store
        store.chunks, store.vectors = chunks, vectors
        store.bm25.add([c.text for c in chunks])
        store._ids = {c.id for c in chunks}
        return store
