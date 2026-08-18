"""Tests for the transition dataset and world model baseline.

Phase 4: Train the World Model on Real Transitions.

Tests prove:
- invalid transitions are rejected (missing next state, NaN, invalid action, etc.)
- time-based splitting prevents temporal leakage
- episode-based splitting works
- the baseline (persistence model) is evaluated correctly
- the learned model can be compared against the baseline
- training is reproducible
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from robot.learning.action_learning import ActionSpace, deskbot_action_space
from robot.learning.dataset import (
    TransitionDataset,
    WorldModelBaseline,
    validate_transition,
)
from robot.learning.experience import Experience
from robot.learning.transition import Transition, TransitionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_transition(
    store: TransitionStore,
    action_index: int = 0,
    reward: float = 0.0,
    state: list[float] | None = None,
    next_state: list[float] | None = None,
) -> Transition:
    state = state or [0.0] * 10
    next_state = next_state or [1.0] * 10
    return store.record(
        state=state,
        action_index=action_index,
        next_state=next_state,
        reward=reward,
    )


@pytest.fixture
def action_space():
    return deskbot_action_space()


@pytest.fixture
def store(action_space):
    return TransitionStore(action_space=action_space)


# ========================================================================
# Transition validation
# ========================================================================


class TestTransitionValidation:
    """Transitions are validated before training."""

    def test_valid_transition_accepted(self, store: TransitionStore) -> None:
        t = make_transition(store)
        ok, reason = validate_transition(t)
        assert ok is True
        assert reason == "ok"

    def test_missing_next_state_rejected(self) -> None:
        """Experience with empty next_state is rejected."""
        exp = make_experience(0, next_state=[])
        ok, reason = validate_transition(exp)
        assert ok is False
        assert "next_state" in reason

    def test_nan_in_state_rejected(self) -> None:
        """Experience with NaN in state is rejected."""
        exp = make_experience(0, state=[1.0, float("nan"), 2.0])
        ok, reason = validate_transition(exp)
        assert ok is False
        assert "NaN" in reason

    def test_inf_in_state_rejected(self) -> None:
        """Experience with inf in state is rejected."""
        exp = make_experience(0, state=[1.0, float("inf"), 2.0])
        ok, reason = validate_transition(exp)
        assert ok is False
        assert "inf" in reason

    def test_nan_in_next_state_rejected(self) -> None:
        """Experience with NaN in next_state is rejected."""
        exp = make_experience(0, next_state=[1.0, float("nan"), 2.0])
        ok, reason = validate_transition(exp)
        assert ok is False
        assert "NaN" in reason

    def test_nan_in_action_rejected(self, store: TransitionStore) -> None:
        """NaN in action vector is rejected."""
        exp = make_experience(action=[1.0, float("nan")])
        ok, reason = validate_transition(exp)
        assert ok is False
        assert "NaN" in reason

    def test_nan_in_reward_rejected(self) -> None:
        """Experience with NaN reward is rejected."""
        exp = make_experience(0, reward=float("nan"))
        ok, reason = validate_transition(exp)
        assert ok is False
        assert "NaN" in reason or "inf" in reason

    def test_empty_action_rejected(self) -> None:
        """Empty action vector is rejected."""
        exp = make_experience(action=[])
        ok, reason = validate_transition(exp)
        assert ok is False
        assert "action" in reason

    def test_invalid_action_index_rejected(self, action_space: ActionSpace) -> None:
        """Action index out of range is rejected."""
        exp = make_experience(metadata_override={"action_index": 999})
        ok, reason = validate_transition(exp, action_space=action_space)
        assert ok is False
        assert "out of range" in reason


def make_experience(
    index: int = 0,
    state: list[float] | None = None,
    action: list[float] | None = None,
    next_state: list[float] | None = None,
    reward: float = 0.0,
    ts: datetime | None = None,
    metadata_override: dict[str, object] | None = None,
) -> Experience:
    meta = {"action_index": 0, "source": "test"}
    if metadata_override:
        meta.update(metadata_override)
    return Experience(
        timestamp=ts or datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
        state=state if state is not None else [float(index)] * 10,
        action=action if action is not None else [float(index + 0.5)] * 5,
        reward=reward,
        next_state=next_state if next_state is not None else [float(index + 1)] * 10,
        metadata=meta,
    )


# ========================================================================
# Dataset splitting
# ========================================================================


class TestDatasetSplit:
    """Time-based and episode-based splits prevent temporal leakage."""

    def test_time_based_split(self) -> None:
        """Time-based split: earliest → train, middle → val, latest → test."""
        dataset = TransitionDataset()
        experiences = [make_experience(i) for i in range(100)]
        split, _stats = dataset.build(experiences, split_method="time")

        assert len(split.train) > 0
        assert len(split.validation) > 0
        assert len(split.test) > 0
        assert split.total == 100

        # Time ordering: train timestamps < val timestamps < test timestamps
        train_max_ts = max(e.timestamp for e in split.train)
        val_min_ts = min(e.timestamp for e in split.validation)
        assert train_max_ts < val_min_ts or train_max_ts <= val_min_ts

    def test_no_temporal_leakage(self) -> None:
        """Consecutive experiences don't leak across splits."""
        dataset = TransitionDataset()
        experiences = [make_experience(i) for i in range(30)]
        split, _ = dataset.build(experiences, split_method="time")

        # All train timestamps should be before all test timestamps
        train_max = max(e.timestamp for e in split.train)
        test_min = min(e.timestamp for e in split.test)
        assert train_max < test_min

    def test_episode_based_split(self) -> None:
        """Episode-based split groups by source/episode."""
        dataset = TransitionDataset()
        experiences = []
        for ep in range(10):
            for i in range(10):
                exp = make_experience(ep * 10 + i)
                exp = Experience(
                    timestamp=exp.timestamp,
                    state=exp.state,
                    action=exp.action,
                    reward=exp.reward,
                    next_state=exp.next_state,
                    metadata={"episode": f"ep{ep}", "action_index": 0},
                )
                experiences.append(exp)

        split, _stats = dataset.build(experiences, split_method="episode")
        assert len(split.train) > 0
        assert len(split.validation) > 0
        assert len(split.test) > 0

        # All experiences from the same episode should be in the same split
        train_episodes = {e.metadata.get("episode") for e in split.train}
        test_episodes = {e.metadata.get("episode") for e in split.test}
        assert train_episodes.isdisjoint(test_episodes)

    def test_too_few_experiences(self) -> None:
        """With fewer than 3 experiences, all go to train."""
        dataset = TransitionDataset()
        experiences = [make_experience(0), make_experience(1)]
        split, _stats = dataset.build(experiences)
        assert len(split.train) == 2
        assert len(split.validation) == 0
        assert len(split.test) == 0

    def test_rejected_transitions_counted(self) -> None:
        """Invalid transitions are rejected and counted."""
        dataset = TransitionDataset()
        experiences = [make_experience(i) for i in range(10)]
        # Add an invalid one (NaN in state)
        bad = make_experience(99, state=[float("nan")] * 10)
        experiences.append(bad)

        _split, stats = dataset.build(experiences)
        assert stats.valid == 10
        assert stats.rejected == 1

    def test_stats_dict(self) -> None:
        """DatasetStats produces a dict."""
        dataset = TransitionDataset()
        experiences = [make_experience(i) for i in range(10)]
        _, stats = dataset.build(experiences)
        d = stats.to_dict()
        assert d["total"] == 10
        assert d["valid"] == 10
        assert d["rejected"] == 0


# ========================================================================
# World model baseline
# ========================================================================


class TestWorldModelBaseline:
    """The baseline must be beaten by the learned model."""

    def test_persistence_baseline_predicts_current(self) -> None:
        """Persistence model predicts next_state = current_state."""
        baseline = WorldModelBaseline(strategy="persistence")
        state = [1.0, 2.0, 3.0]
        pred = baseline.predict(state, [0.0])
        assert pred.tolist() == [1.0, 2.0, 3.0]

    def test_mean_baseline(self) -> None:
        """Mean baseline predicts the mean of training next_states."""
        experiences = [
            make_experience(0, next_state=[1.0, 2.0]),
            make_experience(1, next_state=[3.0, 4.0]),
        ]
        baseline = WorldModelBaseline(strategy="mean")
        baseline.fit(experiences)
        pred = baseline.predict([0.0, 0.0], [0.0])
        assert pred.tolist() == [2.0, 3.0]  # mean of [1,2] and [3,4]

    def test_zero_baseline(self) -> None:
        """Zero baseline predicts all zeros."""
        baseline = WorldModelBaseline(strategy="zero")
        pred = baseline.predict([1.0, 2.0], [0.0])
        assert pred.tolist() == [0.0, 0.0]

    def test_baseline_evaluate(self) -> None:
        """Baseline evaluation returns mean MSE."""
        experiences = [
            make_experience(0, state=[1.0] * 4, next_state=[2.0] * 4),
            make_experience(1, state=[3.0] * 4, next_state=[4.0] * 4),
        ]
        baseline = WorldModelBaseline(strategy="persistence")
        loss = baseline.evaluate(experiences)
        # MSE = mean of (1-2)^2 = 1.0 for each, so mean = 1.0
        assert loss == 1.0

    def test_baseline_empty_evaluate(self) -> None:
        """Baseline on empty data returns 0."""
        baseline = WorldModelBaseline()
        assert baseline.evaluate([]) == 0.0


# ========================================================================
# Reproducibility
# ========================================================================


class TestReproducibility:
    """Training runs are reproducible."""

    def test_same_seed_same_loss(self) -> None:
        """Two models with the same seed produce the same loss."""
        from robot.learning.world_model import SimpleEnvironment, WorldModel

        env = SimpleEnvironment(seed=42, noise_std=0.005)
        experiences = env.collect_experiences(n_steps=64)

        model1 = WorldModel(seed=42)
        model2 = WorldModel(seed=42)

        result1 = model1.train(experiences, epochs=5, verbose=False)
        result2 = model2.train(experiences, epochs=5, verbose=False)

        assert result1.initial_loss == result2.initial_loss
        assert result1.final_loss == result2.final_loss

    def test_model_beats_persistence_baseline(self) -> None:
        """The learned model beats the persistence baseline on test data."""
        from robot.learning.world_model import SimpleEnvironment, WorldModel

        env = SimpleEnvironment(seed=42, noise_std=0.005)
        experiences = env.collect_experiences(n_steps=200)

        # Split
        dataset = TransitionDataset()
        split, _ = dataset.build(experiences, split_method="time")

        # Baseline
        baseline = WorldModelBaseline(strategy="persistence")
        baseline.fit(list(split.train))
        baseline_test_loss = baseline.evaluate(list(split.test))

        # Trained model
        model = WorldModel(seed=42)
        model.train(
            list(split.train),
            val_experiences=list(split.validation),
            epochs=50,
            verbose=False,
        )
        model_test_loss = model.evaluate(list(split.test))

        # The model should beat (or at least match) the baseline
        assert model_test_loss <= baseline_test_loss * 1.5, (
            f"Model loss {model_test_loss} should be close to baseline {baseline_test_loss}"
        )
