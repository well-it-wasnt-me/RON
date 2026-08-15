"""Vector-based semantic memory search.

Extends :class:`Memory` with embedding-based similarity search. When
an embedding function is available (e.g. sentence-transformers), entries
are embedded at add-time and stored alongside their vectors. The
:meth:`search_similar` method returns entries ranked by cosine similarity
to the query embedding.

When no embedding function is available, the system falls back to the
base :class:`Memory` keyword search.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from robot.logging import get_logger

_log = get_logger("ai.vector_memory")


# ---------------------------------------------------------------------------
# Embedding function protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class EmbeddingFn(Protocol):
    """A function that computes an embedding vector from text.

    Implementations may call a local model (sentence-transformers),
    an API (OpenAI embeddings), or any other backend. The vector
    dimension must be consistent across calls.
    """

    def embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for ``text``."""
        ...


class NoOpEmbedding:
    """Fallback embedding that returns a zero vector.

    Used when no embedding backend is configured - all entries get
    the same (zero) embedding and :meth:`search_similar` falls back
    to chronological order.
    """

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        return [0.0] * self._dim


# ---------------------------------------------------------------------------
# Vector entry
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class VectorMemoryEntry:
    """A memory entry with its pre-computed embedding vector."""

    timestamp: datetime
    content: str
    importance: float = 0.5
    tags: tuple[str, ...] = ()
    embedding: tuple[float, ...] = ()

    @property
    def vector(self) -> list[float]:
        return list(self.embedding)


# ---------------------------------------------------------------------------
# Vector memory store
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class VectorMemory:
    """Memory store with vector-based semantic search.

    Entries are embedded at add-time using the configured
    :class:`EmbeddingFn`. If no embedding function is provided,
    a :class:`NoOpEmbedding` is used and :meth:`search_similar`
    returns entries in reverse-chronological order (same as
    :class:`Memory`).

    Keyword search (:meth:`search`) works independently of
    embeddings and is always available.
    """

    entries: deque[VectorMemoryEntry] = field(default_factory=deque)
    capacity: int = 1024
    embedding_fn: EmbeddingFn = field(default_factory=NoOpEmbedding)

    def _fallback_dim(self) -> int:
        """Return the embedding dimension for fallback zero vectors.

        When the embedding function has a _dim attribute set to a
        positive integer, use it.  Otherwise default to 128.
        """
        dim = getattr(self.embedding_fn, "_dim", None)
        return dim if dim is not None and dim > 0 else 128

    def add(self, content: str, importance: float = 0.5, tags: Iterable[str] = ()) -> None:
        """Add an entry, computing and storing its embedding."""
        try:
            vec = self.embedding_fn.embed(content)
        except Exception:
            _log.warning("vector_memory.embed_failed", content=content[:80])
            vec = [0.0] * self._fallback_dim()
        entry = VectorMemoryEntry(
            timestamp=datetime.now(tz=UTC),
            content=content,
            importance=max(0.0, min(1.0, importance)),
            tags=tuple(tags),
            embedding=tuple(vec),
        )
        self.entries.append(entry)
        while len(self.entries) > self.capacity:
            self.entries.popleft()

    def recall(self, limit: int = 10) -> list[VectorMemoryEntry]:
        """Return the most recent ``limit`` entries."""
        return list(self.entries)[-limit:]

    def search(self, query: str) -> list[VectorMemoryEntry]:
        """Keyword search (substring match on content)."""
        q = query.lower()
        return [e for e in self.entries if q in e.content.lower()]

    def search_similar(
        self, query: str, limit: int = 10, min_similarity: float = 0.0
    ) -> list[tuple[VectorMemoryEntry, float]]:
        """Return entries ranked by cosine similarity to ``query``.

        Returns a list of ``(entry, similarity)`` tuples sorted by
        descending similarity. Entries below ``min_similarity`` are
        excluded.

        When using :class:`NoOpEmbedding` (no real embedding backend),
        cosine similarity is meaningless.  In that case, the method
        returns recent entries in reverse chronological order with a
        placeholder similarity of 0.0, making it explicit that no
        semantic search was performed.
        """
        if not self.entries:
            return []

        # Detect NoOpEmbedding fallback: all vectors are zero, so
        # cosine similarity is meaningless. Return recent entries in
        # reverse chronological order instead.
        if isinstance(self.embedding_fn, NoOpEmbedding):
            entries = list(reversed(self.entries))[:limit]
            _log.debug(
                "vector_memory.noop_fallback",
                message="NoOpEmbedding: returning recent entries in chronological order",
                count=len(entries),
            )
            return [(e, 0.0) for e in entries]

        try:
            query_vec = self.embedding_fn.embed(query)
        except Exception:
            _log.warning("vector_memory.query_embed_failed", query=query[:80])
            # Embedding failed; fall back to chronological order.
            entries = list(reversed(self.entries))[:limit]
            return [(e, 0.0) for e in entries]

        results: list[tuple[VectorMemoryEntry, float]] = []
        for entry in self.entries:
            sim = _cosine_similarity(query_vec, list(entry.embedding))
            if sim >= min_similarity:
                results.append((entry, sim))

        results.sort(key=lambda pair: pair[1], reverse=True)
        return results[:limit]

    def clear(self) -> None:
        self.entries.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Sentence-transformers embedding (optional dependency)
# ---------------------------------------------------------------------------
class SentenceTransformerEmbedding:
    """Embedding function backed by ``sentence-transformers``.

    Requires the ``sentence-transformers`` package. Install with::

        pip install sentence-transformers

    Usage::

        from robot.ai.vector_memory import SentenceTransformerEmbedding

        fn = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
        memory = VectorMemory(embedding_fn=fn)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedding. "
                "Install with: pip install sentence-transformers"
            ) from exc
        # Force CPU to avoid CUDA kernel compatibility issues on GPUs
        # with older compute capabilities (e.g. Quadro P520, CC 6.1).
        # The all-MiniLM-L6-v2 model is small enough that CPU inference
        # is fast and avoids the 'no kernel image' error from torch.
        self._model = SentenceTransformer(model_name, device="cpu")
        self._model_name = model_name
        self._dim: int | None = None
        _log.info("vector_memory.model_loaded", model=model_name)

    def embed(self, text: str) -> list[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        vec = [float(x) for x in embedding.tolist()]
        if self._dim is None:
            self._dim = len(vec)
        return vec


__all__ = [
    "EmbeddingFn",
    "NoOpEmbedding",
    "SentenceTransformerEmbedding",
    "VectorMemory",
    "VectorMemoryEntry",
    "_cosine_similarity",
]
