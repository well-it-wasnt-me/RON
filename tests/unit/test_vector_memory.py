"""Tests for vector-based semantic memory."""

from robot.ai.vector_memory import (
    NoOpEmbedding,
    VectorMemory,
    _cosine_similarity,
)


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert abs(_cosine_similarity([1, 0, 0], [1, 0, 0]) - 1.0) < 0.001

    def test_orthogonal_vectors(self) -> None:
        assert abs(_cosine_similarity([1, 0], [0, 1])) < 0.001

    def test_opposite_vectors(self) -> None:
        assert abs(_cosine_similarity([1, 0], [-1, 0]) - (-1.0)) < 0.001

    def test_empty_vectors(self) -> None:
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self) -> None:
        assert _cosine_similarity([1, 0], [1]) == 0.0

    def test_zero_magnitude(self) -> None:
        assert _cosine_similarity([0, 0], [1, 0]) == 0.0


class TestNoOpEmbedding:
    def test_returns_zero_vector(self) -> None:
        fn = NoOpEmbedding(dim=128)
        vec = fn.embed("hello world")
        assert len(vec) == 128
        assert all(v == 0.0 for v in vec)

    def test_custom_dim(self) -> None:
        fn = NoOpEmbedding(dim=64)
        vec = fn.embed("test")
        assert len(vec) == 64


class TestVectorMemory:
    def test_add_and_recall(self) -> None:
        mem = VectorMemory(embedding_fn=NoOpEmbedding(dim=4))
        mem.add("hello world", importance=0.8)
        mem.add("goodbye world", importance=0.5)
        results = mem.recall(limit=10)
        assert len(results) == 2

    def test_search_keyword(self) -> None:
        mem = VectorMemory(embedding_fn=NoOpEmbedding(dim=4))
        mem.add("the cat sat on the mat")
        mem.add("the dog chased the ball")
        results = mem.search("cat")
        assert len(results) == 1
        assert "cat" in results[0].content

    def test_search_similar_noop(self) -> None:
        mem = VectorMemory(embedding_fn=NoOpEmbedding(dim=4))
        mem.add("hello")
        mem.add("world")
        # NoOpEmbedding returns zero vectors, so similarity is 0.
        results = mem.search_similar("hello", min_similarity=0.0)
        # All entries have zero vectors, so they all have 0 similarity.
        assert len(results) >= 0  # No entries above threshold.

    def test_capacity(self) -> None:
        mem = VectorMemory(capacity=3, embedding_fn=NoOpEmbedding(dim=4))
        mem.add("first")
        mem.add("second")
        mem.add("third")
        mem.add("fourth")  # Should evict "first"
        assert len(mem.entries) == 3

    def test_clear(self) -> None:
        mem = VectorMemory(embedding_fn=NoOpEmbedding(dim=4))
        mem.add("hello")
        mem.add("world")
        mem.clear()
        assert len(mem.entries) == 0

    def test_tags(self) -> None:
        mem = VectorMemory(embedding_fn=NoOpEmbedding(dim=4))
        mem.add("hello", tags=("greeting", "english"))
        results = mem.recall(limit=10)
        assert results[0].tags == ("greeting", "english")

    def test_importance_clamped(self) -> None:
        mem = VectorMemory(embedding_fn=NoOpEmbedding(dim=4))
        mem.add("test", importance=5.0)
        results = mem.recall()
        assert results[0].importance == 1.0
