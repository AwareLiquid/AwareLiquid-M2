"""Shared test fixtures.

Tests must run offline, so we never download the real sentence-transformer.
``FakeEncoder`` is a tiny deterministic hashing embedder that satisfies the same
interface (``.dim`` and ``.encode(text, is_query=...)``) the agent depends on.
"""

import torch


class FakeEncoder:
    """Deterministic hashing embedder over character bigrams (no model download)."""

    def __init__(self, dim: int = 64):
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str, is_query: bool = False) -> torch.Tensor:
        vec = torch.zeros(self._dim, dtype=torch.float32)
        text = (text or "").lower()
        for i in range(len(text) - 1):
            bigram = text[i : i + 2]
            vec[hash(bigram) % self._dim] += 1.0
        norm = torch.linalg.vector_norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def as_fn(self, is_query: bool = False):
        return lambda t: self.encode(t, is_query=is_query)
