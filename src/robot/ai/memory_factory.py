"""Factory for creating the appropriate memory backend.

Selects between :class:`Memory` (keyword search) and
:class:`VectorMemory` (semantic search) based on
:class:`VectorMemoryConfig`.
"""

from __future__ import annotations

from robot.ai.memory import Memory
from robot.ai.vector_memory import EmbeddingFn, NoOpEmbedding, VectorMemory
from robot.config import AppSettings
from robot.logging import get_logger

_log = get_logger("ai.memory_factory")


def _create_embedding_fn(vm_cfg: object) -> EmbeddingFn:
    """Create the embedding function based on the vector memory config."""
    backend = getattr(vm_cfg, "backend", "none")
    model_name = getattr(vm_cfg, "model_name", "all-MiniLM-L6-v2")

    if backend == "sentence_transformers":
        try:
            from robot.ai.vector_memory import SentenceTransformerEmbedding

            return SentenceTransformerEmbedding(model_name=model_name)
        except ImportError:
            _log.warning(
                "memory_factory.fallback_noop",
                reason="sentence-transformers not installed",
            )
            return NoOpEmbedding()

    return NoOpEmbedding()


def create_memory(settings: AppSettings) -> Memory | VectorMemory:
    """Create a memory instance based on application settings.

    * When ``settings.vector_memory.enabled`` is ``True``: returns a
      :class:`VectorMemory` configured with the selected embedding backend.
    * When ``settings.vector_memory.enabled`` is ``False`` (default):
      returns a :class:`Memory` with keyword search.

    The embedding backend is selected by ``settings.vector_memory.backend``:

    * ``"none"`` -- :class:`NoOpEmbedding` (zero vectors, falls back to
      chronological order). No extra dependencies required.
    * ``"sentence_transformers"`` -- :class:`SentenceTransformerEmbedding`
      for real semantic similarity. Requires ``sentence-transformers``.
    """
    vm_cfg = settings.vector_memory

    if not vm_cfg.enabled:
        _log.info("memory_factory.keyword_mode", capacity=settings.memory.capacity)
        return Memory(capacity=settings.memory.capacity)

    embedding_fn = _create_embedding_fn(vm_cfg)

    _log.info(
        "memory_factory.vector_mode",
        backend=vm_cfg.backend,
        capacity=vm_cfg.capacity,
    )

    return VectorMemory(
        capacity=vm_cfg.capacity,
        embedding_fn=embedding_fn,
    )


__all__ = ["create_memory"]
