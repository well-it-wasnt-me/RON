"""Tests for experience data structures and memory layers."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from robot.learning.experience import (
    EpisodicMemory,
    Experience,
    InMemoryExperienceStore,
    ReplayBuffer,
    SqliteExperienceStore,
    WorkingMemory,
)
from robot.learning.tensor import Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_experience(
    index: int = 0,
    state_size: int = 4,
    action_size: int = 2,
    reward: float = 1.0,
    ts: datetime | None = None,
) -> Experience:
    """Create a test experience with predictable values."""
    return Experience(
        timestamp=ts or datetime.now(tz=UTC),
        state=[float(index)] * state_size,
        action=[float(index + 0.5)] * action_size,
        reward=reward,
        next_state=[float(index + 1)] * state_size,
        metadata={"index": index, "source": "test"},
    )


# ========================================================================
# Experience dataclass
# ========================================================================


class TestExperience:
    """Tests for the Experience frozen dataclass."""

    def test_creation(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        exp = Experience(
            timestamp=ts,
            state=[1.0, 2.0],
            action=[0.5],
            reward=1.0,
            next_state=[1.1, 2.1],
            metadata={"event": "test"},
        )
        assert exp.timestamp == ts
        assert exp.state == [1.0, 2.0]
        assert exp.action == [0.5]
        assert exp.reward == 1.0
        assert exp.next_state == [1.1, 2.1]
        assert exp.metadata == {"event": "test"}

    def test_frozen(self) -> None:
        exp = _make_experience()
        with pytest.raises(AttributeError):
            exp.reward = 0.0  # type: ignore[misc]

    def test_default_metadata(self) -> None:
        exp = Experience(
            timestamp=datetime.now(tz=UTC),
            state=[1.0],
            action=[0.0],
            reward=0.0,
            next_state=[1.0],
        )
        assert exp.metadata == {}

    def test_state_tensor(self) -> None:
        exp = _make_experience(index=3, state_size=3)
        tensor = exp.state_tensor()
        assert isinstance(tensor, Tensor)
        assert tensor.shape == (3,)
        assert tensor.data.tolist() == pytest.approx([3.0, 3.0, 3.0])

    def test_action_tensor(self) -> None:
        exp = _make_experience(index=1, action_size=2)
        tensor = exp.action_tensor()
        assert isinstance(tensor, Tensor)
        assert tensor.shape == (2,)

    def test_next_state_tensor(self) -> None:
        exp = _make_experience(index=5, state_size=4)
        tensor = exp.next_state_tensor()
        assert isinstance(tensor, Tensor)
        assert tensor.shape == (4,)
        assert tensor.data.tolist() == pytest.approx([6.0, 6.0, 6.0, 6.0])

    def test_serialization_roundtrip(self) -> None:
        ts = datetime(2025, 6, 15, 12, 30, 0, tzinfo=UTC)
        exp = Experience(
            timestamp=ts,
            state=[1.0, 2.0, 3.0],
            action=[0.1, 0.2],
            reward=0.5,
            next_state=[1.1, 2.1, 3.1],
            metadata={"key": "value", "count": 42},
        )
        d = exp.to_dict()
        # Verify JSON-serializable
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        exp2 = Experience.from_dict(loaded)
        assert exp2.timestamp == ts
        assert exp2.state == [1.0, 2.0, 3.0]
        assert exp2.action == [0.1, 0.2]
        assert exp2.reward == 0.5
        assert exp2.next_state == [1.1, 2.1, 3.1]
        assert exp2.metadata == {"key": "value", "count": 42}

    def test_from_dict_iso_timestamp(self) -> None:
        d = {
            "timestamp": "2025-06-15T12:30:00+00:00",
            "state": [1.0],
            "action": [0.0],
            "reward": 0.0,
            "next_state": [1.0],
            "metadata": {},
        }
        exp = Experience.from_dict(d)
        assert exp.timestamp.year == 2025
        assert exp.timestamp.month == 6

    def test_equality(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        exp1 = Experience(ts, [1.0], [0.5], 1.0, [1.1], {})
        exp2 = Experience(ts, [1.0], [0.5], 1.0, [1.1], {})
        assert exp1 == exp2

    def test_inequality(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        exp1 = Experience(ts, [1.0], [0.5], 1.0, [1.1], {})
        exp2 = Experience(ts, [2.0], [0.5], 1.0, [1.1], {})
        assert exp1 != exp2


# ========================================================================
# WorkingMemory
# ========================================================================


class TestWorkingMemory:
    """Tests for the WorkingMemory ring buffer."""

    def test_empty_memory(self) -> None:
        wm = WorkingMemory()
        assert len(wm) == 0
        assert wm.recent() == []

    def test_add_and_recent(self) -> None:
        wm = WorkingMemory()
        exp1 = _make_experience(1)
        exp2 = _make_experience(2)
        wm.add(exp1)
        wm.add(exp2)
        assert len(wm) == 2
        recent = wm.recent(limit=1)
        assert recent == [exp2]

    def test_capacity_eviction(self) -> None:
        wm = WorkingMemory(capacity=3)
        for i in range(10):
            wm.add(_make_experience(i))
        assert len(wm) == 3
        recent = wm.recent(limit=3)
        # Most recent 3 should be indices 7, 8, 9
        assert recent[0].metadata["index"] == 7
        assert recent[1].metadata["index"] == 8
        assert recent[2].metadata["index"] == 9

    def test_clear(self) -> None:
        wm = WorkingMemory()
        wm.add(_make_experience(1))
        wm.add(_make_experience(2))
        assert len(wm) == 2
        wm.clear()
        assert len(wm) == 0

    def test_recent_limit_exceeds_size(self) -> None:
        wm = WorkingMemory()
        wm.add(_make_experience(1))
        recent = wm.recent(limit=10)
        assert len(recent) == 1

    def test_iteration(self) -> None:
        wm = WorkingMemory()
        exps = [_make_experience(i) for i in range(5)]
        for exp in exps:
            wm.add(exp)
        collected = list(wm)
        assert len(collected) == 5
        assert collected[0].metadata["index"] == 0


# ========================================================================
# ReplayBuffer
# ========================================================================


class TestReplayBuffer:
    """Tests for the ReplayBuffer with random sampling."""

    def test_empty_buffer(self) -> None:
        rb = ReplayBuffer(seed=123)
        assert len(rb) == 0
        assert rb.sample(5) == []

    def test_add_and_sample(self) -> None:
        rb = ReplayBuffer(seed=42)
        for i in range(20):
            rb.add(_make_experience(i))
        batch = rb.sample(5)
        assert len(batch) == 5
        # All samples should be valid experiences
        for exp in batch:
            assert isinstance(exp, Experience)
            assert "index" in exp.metadata

    def test_sample_returns_all_if_fewer_than_batch(self) -> None:
        rb = ReplayBuffer(seed=42)
        for i in range(3):
            rb.add(_make_experience(i))
        batch = rb.sample(10)
        assert len(batch) == 3

    def test_deterministic_with_seed(self) -> None:
        rb1 = ReplayBuffer(seed=42)
        rb2 = ReplayBuffer(seed=42)
        for i in range(50):
            exp = _make_experience(i)
            rb1.add(exp)
            rb2.add(exp)
        batch1 = rb1.sample(10)
        batch2 = rb2.sample(10)
        assert batch1 == batch2

    def test_capacity_eviction(self) -> None:
        rb = ReplayBuffer(capacity=5, seed=42)
        for i in range(20):
            rb.add(_make_experience(i))
        assert len(rb) == 5
        recent = list(rb)
        # Should contain the last 5 experiences (indices 15-19)
        indices = [exp.metadata["index"] for exp in recent]
        assert indices == [15, 16, 17, 18, 19]

    def test_clear(self) -> None:
        rb = ReplayBuffer(seed=42)
        for i in range(5):
            rb.add(_make_experience(i))
        assert len(rb) == 5
        rb.clear()
        assert len(rb) == 0

    def test_no_replacement_in_sample(self) -> None:
        rb = ReplayBuffer(seed=99)
        for i in range(100):
            rb.add(_make_experience(i))
        batch = rb.sample(50)
        ids = [id(e) for e in batch]
        assert len(ids) == len(set(ids)), "Sample should not contain duplicates"


# ========================================================================
# SqliteExperienceStore
# ========================================================================


class TestSqliteExperienceStore:
    """Tests for SQLite-backed experience persistence."""

    def test_save_and_load(self) -> None:
        store = SqliteExperienceStore(db_path=":memory:")
        exp = _make_experience(0)
        row_id = store.save(exp)
        assert row_id == 1
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].state == [0.0, 0.0, 0.0, 0.0]
        assert loaded[0].action == [0.5, 0.5]
        assert loaded[0].reward == 1.0
        store.close()

    def test_save_batch(self) -> None:
        store = SqliteExperienceStore(db_path=":memory:")
        exps = [_make_experience(i) for i in range(5)]
        ids = store.save_batch(exps)
        assert len(ids) == 5
        assert ids == [1, 2, 3, 4, 5]
        assert store.count() == 5
        store.close()

    def test_load_recent(self) -> None:
        store = SqliteExperienceStore(db_path=":memory:")
        base_time = datetime(2025, 1, 1, tzinfo=UTC)
        for i in range(10):
            exp = _make_experience(i, ts=base_time + timedelta(hours=i))
            store.save(exp)
        recent = store.load_recent(limit=3)
        assert len(recent) == 3
        # Most recent should be indices 7, 8, 9 (in descending order)
        indices = [exp.metadata["index"] for exp in recent]
        assert indices == [9, 8, 7]
        store.close()

    def test_delete_before(self) -> None:
        store = SqliteExperienceStore(db_path=":memory:")
        base_time = datetime(2025, 1, 1, tzinfo=UTC)
        for i in range(5):
            exp = _make_experience(i, ts=base_time + timedelta(hours=i))
            store.save(exp)
        cutoff = base_time + timedelta(hours=2)
        deleted = store.delete_before(cutoff)
        assert deleted == 2
        assert store.count() == 3
        store.close()

    def test_count_empty(self) -> None:
        store = SqliteExperienceStore(db_path=":memory:")
        assert store.count() == 0
        store.close()

    def test_roundtrip_metadata(self) -> None:
        store = SqliteExperienceStore(db_path=":memory:")
        exp = Experience(
            timestamp=datetime(2025, 6, 1, tzinfo=UTC),
            state=[1.0, 2.0],
            action=[0.5],
            reward=0.8,
            next_state=[1.1, 2.1],
            metadata={"source": "test", "tags": ["a", "b"]},
        )
        store.save(exp)
        loaded = store.load_all()
        assert loaded[0].metadata["source"] == "test"
        assert loaded[0].metadata["tags"] == ["a", "b"]
        store.close()

    def test_file_persistence(self, tmp_path: Path) -> None:
        """Experiences survive closing and reopening the database."""
        db_path = tmp_path / "test_experiences.db"
        exp = _make_experience(42, ts=datetime(2025, 3, 1, tzinfo=UTC))

        # Save
        store = SqliteExperienceStore(db_path=db_path)
        store.save(exp)
        assert store.count() == 1
        store.close()

        # Reload
        store2 = SqliteExperienceStore(db_path=db_path)
        loaded = store2.load_all()
        assert len(loaded) == 1
        assert loaded[0].metadata["index"] == 42
        store2.close()


# ========================================================================
# InMemoryExperienceStore
# ========================================================================


class TestInMemoryExperienceStore:
    """Tests for in-memory experience store."""

    def test_save_and_load(self) -> None:
        store = InMemoryExperienceStore()
        exp = _make_experience(0)
        row_id = store.save(exp)
        assert row_id == 1
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].state == exp.state

    def test_save_batch(self) -> None:
        store = InMemoryExperienceStore()
        exps = [_make_experience(i) for i in range(3)]
        ids = store.save_batch(exps)
        assert len(ids) == 3
        assert store.count() == 3

    def test_load_recent(self) -> None:
        store = InMemoryExperienceStore()
        for i in range(10):
            store.save(_make_experience(i))
        recent = store.load_recent(limit=3)
        assert len(recent) == 3

    def test_delete_before(self) -> None:
        store = InMemoryExperienceStore()
        base_time = datetime(2025, 1, 1, tzinfo=UTC)
        exps = [_make_experience(i, ts=base_time + timedelta(hours=i)) for i in range(5)]
        for exp in exps:
            store.save(exp)
        cutoff = base_time + timedelta(hours=2)
        deleted = store.delete_before(cutoff)
        assert deleted == 2
        assert store.count() == 3

    def test_close_is_noop(self) -> None:
        store = InMemoryExperienceStore()
        store.close()  # Should not raise


# ========================================================================
# EpisodicMemory
# ========================================================================


class TestEpisodicMemory:
    """Tests for EpisodicMemory combining buffer and persistence."""

    def test_add_and_recent(self) -> None:
        store = InMemoryExperienceStore()
        mem = EpisodicMemory(store=store)
        for i in range(5):
            mem.add(_make_experience(i))
        recent = mem.recent(limit=3)
        assert len(recent) == 3
        # Most recent should be last added
        assert recent[-1].metadata["index"] == 4

    def test_capacity_eviction(self) -> None:
        store = InMemoryExperienceStore()
        mem = EpisodicMemory(store=store, capacity=3)
        for i in range(10):
            mem.add(_make_experience(i))
        assert len(mem) == 3
        # Most recent 3 should be preserved
        recent = mem.recent(limit=3)
        indices = [exp.metadata["index"] for exp in recent]
        assert indices == [7, 8, 9]

    def test_persisted_on_add(self) -> None:
        store = InMemoryExperienceStore()
        mem = EpisodicMemory(store=store)
        exp = _make_experience(1)
        mem.add(exp)
        # Should be persisted in the store
        assert store.count() == 1

    def test_load_from_store(self) -> None:
        store = InMemoryExperienceStore()
        # Pre-populate the store
        exps = [_make_experience(i) for i in range(5)]
        store.save_batch(exps)
        # Create EpisodicMemory and load from store
        mem = EpisodicMemory(store=store, max_load=5)
        mem.load_from_store()
        assert len(mem) == 5

    def test_restart_reload(self, tmp_path: Path) -> None:
        """Experiences survive a simulated restart (close + reload)."""
        db_path = tmp_path / "episodic_test.db"
        store1 = SqliteExperienceStore(db_path=db_path)
        mem1 = EpisodicMemory(store=store1, capacity=100)
        for i in range(5):
            mem1.add(_make_experience(i, ts=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i)))
        assert len(mem1) == 5
        store1.close()

        # Simulate restart
        store2 = SqliteExperienceStore(db_path=db_path)
        mem2 = EpisodicMemory(store=store2, capacity=100, max_load=10)
        mem2.load_from_store()
        assert len(mem2) == 5
        recent = mem2.recent(limit=5)
        # All 5 experiences should be present
        indices = sorted([exp.metadata["index"] for exp in recent])
        assert indices == [0, 1, 2, 3, 4]
        store2.close()

    def test_sample(self) -> None:
        store = InMemoryExperienceStore()
        mem = EpisodicMemory(store=store)
        for i in range(20):
            mem.add(_make_experience(i))
        batch = mem.sample(batch_size=5, seed=42)
        assert len(batch) == 5
        for exp in batch:
            assert isinstance(exp, Experience)

    def test_sample_empty(self) -> None:
        store = InMemoryExperienceStore()
        mem = EpisodicMemory(store=store)
        mem.load_from_store()
        batch = mem.sample(batch_size=5, seed=42)
        assert batch == []

    def test_clear_buffer_not_store(self) -> None:
        store = InMemoryExperienceStore()
        mem = EpisodicMemory(store=store)
        mem.add(_make_experience(1))
        mem.add(_make_experience(2))
        assert store.count() == 2
        mem.clear()
        assert len(mem) == 0
        # Store still has the data
        assert store.count() == 2


# ========================================================================
# Integration: observe -> act -> observe -> store -> restart -> reload
# ========================================================================


class TestExperienceIntegration:
    """End-to-end integration test matching the acceptance criteria."""

    def test_observe_act_store_reload(self, tmp_path: Path) -> None:
        """Verify: observe -> act -> observe result -> store experience -> restart -> load experience."""
        db_path = tmp_path / "integration_test.db"

        # Phase 1: Create experiences
        store1 = SqliteExperienceStore(db_path=db_path)
        mem1 = EpisodicMemory(store=store1, capacity=100)

        base_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)

        # Observe state
        state_1 = [0.1, 0.2, 0.3, 0.4]
        # Act
        action_1 = [0.5, 0.6]
        # Observe result
        next_state_1 = [0.15, 0.25, 0.35, 0.45]
        # Store experience
        exp1 = Experience(
            timestamp=base_time,
            state=state_1,
            action=action_1,
            reward=1.0,
            next_state=next_state_1,
            metadata={"source": "integration_test"},
        )
        mem1.add(exp1)

        # Second experience
        state_2 = next_state_1
        action_2 = [0.7, 0.8]
        next_state_2 = [0.2, 0.3, 0.4, 0.5]
        exp2 = Experience(
            timestamp=base_time + timedelta(seconds=1),
            state=state_2,
            action=action_2,
            reward=0.5,
            next_state=next_state_2,
            metadata={"source": "integration_test"},
        )
        mem1.add(exp2)

        assert len(mem1) == 2
        assert store1.count() == 2
        store1.close()

        # Phase 2: Restart and reload
        store2 = SqliteExperienceStore(db_path=db_path)
        mem2 = EpisodicMemory(store=store2, capacity=100, max_load=10)
        mem2.load_from_store()

        # Experiences must survive restart
        assert len(mem2) == 2
        loaded = mem2.recent(limit=2)
        assert len(loaded) == 2

        # Verify data integrity
        exp_loaded = loaded[0]
        assert exp_loaded.state == state_1
        assert exp_loaded.action == action_1
        assert exp_loaded.reward == 1.0
        assert exp_loaded.next_state == next_state_1

        store2.close()

    def test_working_memory_to_replay_buffer(self) -> None:
        """Experiences flow from working memory to replay buffer."""
        wm = WorkingMemory(capacity=100)
        rb = ReplayBuffer(capacity=100, seed=42)

        for i in range(20):
            exp = _make_experience(i)
            wm.add(exp)
            rb.add(exp)

        assert len(wm) == 20
        assert len(rb) == 20

        # Sample from replay buffer
        batch = rb.sample(5)
        assert len(batch) == 5

        # Working memory recent retrieval
        recent = wm.recent(limit=5)
        assert len(recent) == 5
        assert recent[-1].metadata["index"] == 19

    def test_full_memory_pipeline(self) -> None:
        """Working memory -> replay buffer -> episodic memory pipeline."""
        store = InMemoryExperienceStore()
        wm = WorkingMemory(capacity=100)
        rb = ReplayBuffer(capacity=100, seed=42)
        em = EpisodicMemory(store=store, capacity=100)

        for i in range(30):
            exp = _make_experience(i)
            wm.add(exp)
            rb.add(exp)
            em.add(exp)

        assert len(wm) == 30
        assert len(rb) == 30
        assert len(em) == 30
        assert store.count() == 30

        # Replay buffer can sample
        batch = rb.sample(10)
        assert len(batch) == 10

        # Episodic memory can sample
        batch_em = em.sample(10, seed=42)
        assert len(batch_em) == 10

        # Working memory recent access
        recent = wm.recent(limit=5)
        assert len(recent) == 5
