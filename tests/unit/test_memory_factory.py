"""Tests for the memory factory."""

from __future__ import annotations

import os

from robot.ai.memory import Memory
from robot.ai.memory_factory import create_memory
from robot.ai.vector_memory import NoOpEmbedding, VectorMemory
from robot.config import AppSettings, VectorMemoryConfig


def _make_settings(
    *,
    memory_enabled: bool = True,
    memory_capacity: int = 1024,
    vm_enabled: bool = False,
    vm_backend: str = "none",
    vm_capacity: int = 2048,
) -> AppSettings:
    """Build an AppSettings with the given overrides."""
    settings = AppSettings()
    settings.memory.enabled = memory_enabled
    settings.memory.capacity = memory_capacity
    settings.vector_memory.enabled = vm_enabled
    settings.vector_memory.backend = vm_backend  # type: ignore[assignment]
    settings.vector_memory.capacity = vm_capacity
    return settings


class TestCreateMemory:
    def test_returns_memory_when_vector_disabled(self) -> None:
        settings = _make_settings(vm_enabled=False)
        mem = create_memory(settings)
        assert isinstance(mem, Memory)

    def test_returns_vector_memory_when_enabled(self) -> None:
        settings = _make_settings(vm_enabled=True, vm_backend="none")
        mem = create_memory(settings)
        assert isinstance(mem, VectorMemory)

    def test_vector_memory_uses_noop_by_default(self) -> None:
        settings = _make_settings(vm_enabled=True, vm_backend="none")
        mem = create_memory(settings)
        assert isinstance(mem, VectorMemory)
        assert isinstance(mem.embedding_fn, NoOpEmbedding)

    def test_vector_memory_capacity_from_config(self) -> None:
        settings = _make_settings(vm_enabled=True, vm_backend="none", vm_capacity=512)
        mem = create_memory(settings)
        assert isinstance(mem, VectorMemory)
        assert mem.capacity == 512

    def test_memory_capacity_from_config(self) -> None:
        settings = _make_settings(vm_enabled=False, memory_capacity=256)
        mem = create_memory(settings)
        assert isinstance(mem, Memory)
        assert mem.capacity == 256

    def test_returns_none_when_memory_disabled(self) -> None:
        """When memory.enabled is False, create_memory returns Memory
        but the calling code in app.py wraps it in `if settings.memory.enabled`."""
        settings = _make_settings(memory_enabled=False)
        # create_memory itself still returns a Memory; the app.py checks enabled.
        mem = create_memory(settings)
        assert isinstance(mem, Memory)

    def test_vector_memory_noop_embedding_dimension(self) -> None:
        settings = _make_settings(vm_enabled=True, vm_backend="none")
        mem = create_memory(settings)
        assert isinstance(mem, VectorMemory)
        # Add and recall should work with NoOpEmbedding
        mem.add("hello world", importance=0.8)
        results = mem.recall(limit=5)
        assert len(results) == 1
        assert results[0].content == "hello world"

    def test_vector_memory_search_similar_noop(self) -> None:
        """With NoOpEmbedding, search_similar returns entries in reverse
        chronological order (all zero vectors have cosine similarity 0)."""
        settings = _make_settings(vm_enabled=True, vm_backend="none")
        mem = create_memory(settings)
        assert isinstance(mem, VectorMemory)
        mem.add("first entry")
        mem.add("second entry")
        mem.add("third entry")
        # With min_similarity=0.0, all entries are returned
        results = mem.search_similar("test", limit=10, min_similarity=0.0)
        # NoOpEmbedding produces zero vectors => cosine similarity = 0
        assert len(results) == 3

    def test_vector_memory_keyword_search_works(self) -> None:
        """Keyword search should work regardless of embedding backend."""
        settings = _make_settings(vm_enabled=True, vm_backend="none")
        mem = create_memory(settings)
        assert isinstance(mem, VectorMemory)
        mem.add("the cat sat on the mat")
        mem.add("the dog chased the ball")
        results = mem.search("cat")
        assert len(results) == 1
        assert "cat" in results[0].content


class TestVectorMemoryConfig:
    def test_defaults(self) -> None:
        cfg = VectorMemoryConfig()
        assert cfg.enabled is False
        assert cfg.backend == "none"
        assert cfg.model_name == "all-MiniLM-L6-v2"
        assert cfg.capacity == 2048
        assert cfg.similarity_threshold == 0.3
        assert cfg.recall_limit == 5

    def test_env_override(self) -> None:
        """VectorMemoryConfig should load from DESKBOT_VECTOR_MEMORY__* env vars."""
        env = {
            "DESKBOT_VECTOR_MEMORY__ENABLED": "true",
            "DESKBOT_VECTOR_MEMORY__BACKEND": "none",
            "DESKBOT_VECTOR_MEMORY__CAPACITY": "512",
        }
        original = {}
        for key, value in env.items():
            original[key] = os.environ.get(key)
            os.environ[key] = value
        try:
            cfg = VectorMemoryConfig()
            assert cfg.enabled is True
            assert cfg.backend == "none"
            assert cfg.capacity == 512
        finally:
            for key, value in original.items():  # type: ignore[assignment]
                if value is None:
                    os.environ.pop(key, None)  # type: ignore[unreachable]
                else:
                    os.environ[key] = value
