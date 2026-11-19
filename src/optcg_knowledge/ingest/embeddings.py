"""Embedding interfaces, vector serialization, and deterministic offline embedder."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"\[[^\]]+\]|[a-z0-9]+(?:-[a-z0-9]+)*")


@runtime_checkable
class Embedder(Protocol):
    """Protocol implemented by local sentence-transformers and test embedders."""

    @property
    def model_name(self) -> str:
        """Identifier of the underlying embedding model."""
        ...

    @property
    def dimension(self) -> int:
        """Vector dimensionality produced by embed()."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of texts into unit-normalized float vectors."""
        ...


def serialize_vector(vector: Sequence[float]) -> bytes:
    """Serialize a float sequence into sqlite-vec little-endian float32 bytes."""
    return struct.pack(f"<{len(vector)}f", *vector)


class HashEmbedder:
    """Lightweight deterministic embedder for fast offline testing."""

    def __init__(self, *, dimension: int = 64, model_name: str = "test/hash-embedder") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            vec[0] = 1.0
            return vec

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]
