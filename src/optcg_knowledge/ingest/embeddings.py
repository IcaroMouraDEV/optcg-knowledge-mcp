"""Embedding interfaces, vector serialization, local ST embedder, and offline test embedder."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import struct
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from typing import Any, Protocol, runtime_checkable

from optcg_knowledge.config import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_FP16,
    DEFAULT_EMBEDDING_MODEL,
    resolve_embedding_dim,
)

logger = logging.getLogger(__name__)

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
        """Encode a batch of document texts into unit-normalized float vectors."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Encode a single search query into a unit-normalized float vector."""
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

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

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


def _requires_e5_prefix(model_name: str) -> bool:
    lowered = model_name.lower()
    return "multilingual-e5" in lowered or "/e5-" in lowered or lowered.startswith("e5-")


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder with CUDA FP16 support and CPU fallback.

    Imports ``torch`` and ``sentence_transformers`` lazily so CLI commands and offline
    tests work even when optional ML dependencies are not installed.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = DEFAULT_EMBEDDING_DEVICE,
        use_fp16: bool = DEFAULT_EMBEDDING_FP16,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        dimension: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._model_name = model_name
        self._requested_device = device
        self._resolved_device: str | None = None
        self._use_fp16 = use_fp16
        self._batch_size = batch_size
        self._dimension = resolve_embedding_dim(model_name, override=dimension)
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def resolved_device(self) -> str | None:
        return self._resolved_device

    def load(self) -> Any:
        """Load and cache the underlying SentenceTransformer model."""
        if self._model is not None:
            return self._model

        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require the optional 'embeddings' dependencies. "
                "Install them with: uv sync --extra embeddings"
            ) from exc

        device = self._resolve_device(torch)
        self._resolved_device = device

        model_kwargs: dict[str, Any] = {}
        dtype_label = "fp32"
        if self._use_fp16 and device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16
            dtype_label = "fp16"

        logger.info(
            "Loading local embedding model %s on %s (%s, batch_size=%d)",
            self._model_name,
            device,
            dtype_label,
            self._batch_size,
        )
        # Redirect stdout to stderr during model load so third-party HF/transformers warnings
        # or progress bars can never corrupt MCP stdio JSON-RPC frames.
        with redirect_stdout(sys.stderr):
            try:
                model = SentenceTransformer(
                    self._model_name,
                    device=device,
                    model_kwargs=model_kwargs or None,
                    local_files_only=True,
                )
            except Exception:
                model = SentenceTransformer(
                    self._model_name,
                    device=device,
                    model_kwargs=model_kwargs or None,
                )

        get_dim = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension", None
        )
        detected_dim = get_dim() if callable(get_dim) else None
        if isinstance(detected_dim, int) and detected_dim > 0:
            self._dimension = detected_dim

        self._model = model
        return model

    def _resolve_device(self, torch_module: Any) -> str:
        requested = self._requested_device.strip().lower()
        if requested == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch_module.cuda.is_available():
            logger.warning("CUDA requested (%s) but unavailable; falling back to CPU", requested)
            return "cpu"
        return requested

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode document passages into unit-normalized float vectors."""
        if not texts:
            return []
        prepared = (
            [f"passage: {text}" for text in texts]
            if _requires_e5_prefix(self._model_name)
            else list(texts)
        )
        return self._encode(prepared)

    def embed_query(self, text: str) -> list[float]:
        """Encode a search query into a unit-normalized float vector."""
        prepared = f"query: {text}" if _requires_e5_prefix(self._model_name) else text
        return self._encode([prepared])[0]

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self.load()
        with redirect_stdout(sys.stderr):
            raw_vectors = model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [[float(x) for x in row] for row in raw_vectors]
