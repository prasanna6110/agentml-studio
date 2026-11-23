"""Text embedders. The default needs no downloads; dense transformer embeddings are optional."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Fast, stateless sparse→dense embedding using word + character n-gram feature hashing.

    Captures lexical and sub-word similarity, works offline, and needs no fitting, which makes it
    a solid default for small knowledge bases and for tests.
    """

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim
        self._word = HashingVectorizer(n_features=dim, ngram_range=(1, 2), alternate_sign=False,
                                       norm=None, stop_words="english")
        self._char = HashingVectorizer(n_features=dim, analyzer="char_wb", ngram_range=(3, 5),
                                       alternate_sign=False, norm=None)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        w = self._word.transform(texts).toarray()
        c = self._char.transform(texts).toarray()
        v = np.log1p(w) * 1.0 + np.log1p(c) * 0.35
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return (v / np.where(norms == 0, 1, norms)).astype(np.float32)


class SentenceTransformerEmbedder:
    """Dense semantic embeddings (e.g. ``all-MiniLM-L6-v2``). Requires ``sentence-transformers``."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def get_embedder(name: str = "hashing") -> Embedder:
    if name in {"hashing", "default"}:
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(name if "/" in name else "sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        return HashingEmbedder()
