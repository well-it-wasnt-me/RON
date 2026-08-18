"""Tests for controlled online learning.

Phase 9: Controlled Online Learning.

Tests prove:
- the monitor tracks all required metrics
- exploration is constrained (no unrestricted epsilon-greedy on robot)
- the replay buffer is warmed from persistent storage
- training metrics are recorded correctly
- action distribution is tracked
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from robot.learning.experience import (
    Experience,
    InMemoryExperienceStore,
    ReplayBuffer,
    SqliteExperienceStore,
)
from robot.learning.online_learning import (
    ConstrainedExploration,
    OnlineLearningMonitor,
    ReplayWarmer,
)

# ========================================================================
# Online learning monitor
# ========================================================================


class TestOnlineLearningMonitor:
    """The monitor tracks all required metrics."""

    def test_record_action(self) -> None:
        m = OnlineLearningMonitor()
        m.record_action(0)
        m.record_action(0)
        m.record_action(2)
        dist = m.action_distribution
        assert dist[0] == pytest.approx(2 / 3)
        assert dist[2] == pytest.approx(1 / 3)

    def test_record_safety_rejection(self) -> None:
        m = OnlineLearningMonitor()
        m.record_safety_rejection()
        m.record_safety_rejection()
        assert m.safety_rejections == 2

    def test_record_fallback(self) -> None:
        m = OnlineLearningMonitor()
        m.record_fallback()
        assert m.fallback_count == 1

    def test_record_sensor_dropout(self) -> None:
        m = OnlineLearningMonitor()
        m.record_sensor_dropout()
        assert m.sensor_dropout_count == 1

    def test_record_inference_latency(self) -> None:
        m = OnlineLearningMonitor()
        m.record_inference_latency(10.0)
        m.record_inference_latency(20.0)
        assert m.inference_latency_ms == pytest.approx(15.0)

    def test_record_model_load_failure(self) -> None:
        m = OnlineLearningMonitor()
        m.record_model_load_failure()
        assert m.model_load_failures == 1

    def test_record_training(self) -> None:
        m = OnlineLearningMonitor()
        m.record_training(loss=0.5, val_loss=0.6)
        assert m.training_loss == 0.5
        assert m.validation_loss == 0.6

    def test_record_reward(self) -> None:
        m = OnlineLearningMonitor()
        m.record_reward(1.0)
        m.record_reward(0.5)
        assert m.total_reward == 1.5

    def test_to_dict(self) -> None:
        m = OnlineLearningMonitor(model_version=5, replay_size=100)
        m.record_action(0)
        d = m.to_dict()
        assert d["model_version"] == 5
        assert d["replay_size"] == 100
        assert "action_distribution" in d
        assert "safety_rejections" in d
        assert "fallback_count" in d
        assert "inference_latency_ms" in d
        assert "model_load_failures" in d
        assert "training_loss" in d
        assert "validation_loss" in d


# ========================================================================
# Constrained exploration
# ========================================================================


class TestConstrainedExploration:
    """Exploration is constrained — no unrestricted epsilon-greedy on the robot."""

    def test_allowed_actions(self) -> None:
        e = ConstrainedExploration(allowed_action_indices={0, 1, 2})
        assert e.is_action_allowed(0)
        assert e.is_action_allowed(1)
        assert not e.is_action_allowed(5)

    def test_no_restriction(self) -> None:
        e = ConstrainedExploration()
        assert e.is_action_allowed(99)

    def test_clamp_rate(self) -> None:
        e = ConstrainedExploration(max_exploration_rate=0.1, min_exploration_rate=0.01)
        assert e.clamp_rate(0.5) == 0.1
        assert e.clamp_rate(0.001) == 0.01
        assert e.clamp_rate(0.05) == 0.05


# ========================================================================
# Replay warmer
# ========================================================================


class TestReplayWarmer:
    """The replay buffer is warmed from persistent storage after reboot."""

    def test_warm_from_in_memory_store(self) -> None:
        store = InMemoryExperienceStore()
        for i in range(10):
            store.save(
                Experience(
                    timestamp=datetime.now(tz=UTC),
                    state=[float(i)] * 4,
                    action=[0.0],
                    reward=float(i),
                    next_state=[float(i + 1)] * 4,
                    metadata={},
                )
            )
        warmer = ReplayWarmer(store=store, max_warm=10)
        rb = ReplayBuffer(capacity=100, seed=42)
        count = warmer.warm(rb)
        assert count == 10
        assert len(rb) == 10

    def test_warm_from_sqlite(self, tmp_path: Path) -> None:
        db = tmp_path / "experiences.db"
        store = SqliteExperienceStore(db_path=db)
        for i in range(5):
            store.save(
                Experience(
                    timestamp=datetime.now(tz=UTC),
                    state=[float(i)] * 4,
                    action=[0.0],
                    reward=float(i),
                    next_state=[float(i + 1)] * 4,
                    metadata={},
                )
            )
        store.close()

        # Simulate restart
        store2 = SqliteExperienceStore(db_path=db)
        warmer = ReplayWarmer(store=store2, max_warm=5)
        rb = ReplayBuffer(capacity=100, seed=42)
        count = warmer.warm(rb)
        assert count == 5
        assert len(rb) == 5
        store2.close()

    def test_warm_empty_store(self) -> None:
        store = InMemoryExperienceStore()
        warmer = ReplayWarmer(store=store, max_warm=10)
        rb = ReplayBuffer(capacity=100, seed=42)
        count = warmer.warm(rb)
        assert count == 0
        assert len(rb) == 0

    def test_warm_respects_max(self) -> None:
        store = InMemoryExperienceStore()
        for i in range(100):
            store.save(
                Experience(
                    timestamp=datetime.now(tz=UTC),
                    state=[float(i)] * 4,
                    action=[0.0],
                    reward=float(i),
                    next_state=[float(i + 1)] * 4,
                    metadata={},
                )
            )
        warmer = ReplayWarmer(store=store, max_warm=10)
        rb = ReplayBuffer(capacity=100, seed=42)
        count = warmer.warm(rb)
        assert count == 10
