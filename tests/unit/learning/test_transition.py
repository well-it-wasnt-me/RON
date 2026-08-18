"""Tests for the TransitionStore transition lifecycle.

These tests prove the Phase 1 acceptance criteria:

- no action means no completed transition
- failed action execution is recorded correctly
- next_state is captured after execution
- action ID belongs to the configured action space
- timestamps are monotonic
- malformed transitions are rejected

Definition of done: a stored transition can be reconstructed as:

    At T0: observation = X
    At T0: policy selected action = Y
    Robot executed Y
    At T1: observation = Z
    Reward = R
"""

from __future__ import annotations

import time

import pytest

from robot.learning.action_learning import ActionSpace, deskbot_action_space
from robot.learning.experience import Experience
from robot.learning.transition import (
    PendingTransition,
    Transition,
    TransitionError,
    TransitionStore,
)

# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def action_space():
    return deskbot_action_space()


@pytest.fixture
def store(action_space):
    return TransitionStore(action_space=action_space)


def _state(seed: int = 0, size: int = 10) -> list[float]:
    """Create a deterministic test state vector."""
    return [float(seed + i) / 100.0 for i in range(size)]


# ========================================================================
# Begin: opening a transition
# ========================================================================


class TestBegin:
    """Tests for TransitionStore.begin()."""

    def test_begin_returns_pending(self, store: TransitionStore) -> None:
        """begin() returns a PendingTransition with correct action identity."""
        state = _state(0)
        pending = store.begin(state=state, action_index=2)
        assert isinstance(pending, PendingTransition)
        assert pending.action_index == 2
        assert pending.action_name == "look_center"
        assert pending.state == state
        assert pending.transition_id != ""
        assert pending.execution_id != ""
        assert pending.policy_version == "deterministic"
        assert pending.start_timestamp_ns > 0
        assert store.pending_count == 1

    def test_begin_captures_state_copy(self, store: TransitionStore) -> None:
        """begin() copies the state so later mutations don't affect it."""
        state = _state(0)
        pending = store.begin(state=state, action_index=0)
        state[0] = 999.0  # mutate original
        assert pending.state[0] == 0.0  # pending has the original

    def test_begin_invalid_action_index(self, store: TransitionStore) -> None:
        """Out-of-range action index is rejected."""
        with pytest.raises(TransitionError, match="out of range"):
            store.begin(state=_state(0), action_index=999)

    def test_begin_negative_action_index(self, store: TransitionStore) -> None:
        """Negative action index is rejected."""
        with pytest.raises(TransitionError, match="out of range"):
            store.begin(state=_state(0), action_index=-1)

    def test_begin_empty_state_rejected(self, store: TransitionStore) -> None:
        """Empty state is rejected."""
        with pytest.raises(TransitionError, match="state must not be empty"):
            store.begin(state=[], action_index=0)

    def test_begin_nan_state_rejected(self, store: TransitionStore) -> None:
        """NaN in state is rejected."""
        state = [1.0, float("nan"), 2.0]
        with pytest.raises(TransitionError, match="NaN"):
            store.begin(state=state, action_index=0)

    def test_begin_inf_state_rejected(self, store: TransitionStore) -> None:
        """Inf in state is rejected."""
        state = [1.0, float("inf"), 2.0]
        with pytest.raises(TransitionError, match="inf"):
            store.begin(state=state, action_index=0)

    def test_begin_with_custom_execution_id(self, store: TransitionStore) -> None:
        """Custom execution_id is used."""
        pending = store.begin(state=_state(0), action_index=0, execution_id="exec-42")
        assert pending.execution_id == "exec-42"

    def test_begin_with_custom_policy_version(self, store: TransitionStore) -> None:
        """Custom policy_version is used."""
        pending = store.begin(state=_state(0), action_index=0, policy_version="v2.0.0")
        assert pending.policy_version == "v2.0.0"

    def test_all_action_indices_valid(
        self, store: TransitionStore, action_space: ActionSpace
    ) -> None:
        """Every action index in the action space is valid for begin()."""
        for i in range(action_space.size):
            pending = store.begin(state=_state(i), action_index=i)
            assert pending.action_index == i
            assert pending.action_name == action_space.get(i).name


# ========================================================================
# Complete: closing a transition
# ========================================================================


class TestComplete:
    """Tests for PendingTransition.complete() / TransitionStore._complete()."""

    def test_complete_produces_transition(self, store: TransitionStore) -> None:
        """complete() returns a Transition with all fields populated."""
        pending = store.begin(state=_state(0), action_index=2)
        transition = pending.complete(
            next_state=_state(1),
            reward=0.5,
            done=False,
        )
        assert isinstance(transition, Transition)
        assert transition.action_index == 2
        assert transition.action_name == "look_center"
        assert transition.reward == 0.5
        assert transition.done is False
        assert transition.state == _state(0)
        assert transition.next_state == _state(1)
        assert store.pending_count == 0

    def test_no_action_means_no_completed_transition(self, store: TransitionStore) -> None:
        """Without calling complete, no transition is stored."""
        store.begin(state=_state(0), action_index=0)
        assert store.pending_count == 1
        # No callback was invoked — nothing stored

    def test_complete_captures_next_state_after_execution(self, store: TransitionStore) -> None:
        """next_state reflects the post-execution observation."""
        state_t = _state(0)
        state_t1 = _state(99)
        pending = store.begin(state=state_t, action_index=0)
        transition = pending.complete(next_state=state_t1, reward=1.0)
        assert transition.state == state_t
        assert transition.next_state == state_t1
        assert transition.state != transition.next_state

    def test_failed_action_execution_recorded(self, store: TransitionStore) -> None:
        """A failed execution is recorded with success=False and a reason."""
        pending = store.begin(state=_state(0), action_index=0)
        transition = pending.complete(
            next_state=_state(1),
            reward=-1.0,
            execution_success=False,
            execution_failure_reason="servo jammed",
        )
        assert transition.execution_success is False
        assert transition.execution_failure_reason == "servo jammed"

    def test_successful_execution_recorded(self, store: TransitionStore) -> None:
        """A successful execution has success=True."""
        pending = store.begin(state=_state(0), action_index=0)
        transition = pending.complete(next_state=_state(1), reward=0.0)
        assert transition.execution_success is True
        assert transition.execution_failure_reason == ""

    def test_complete_twice_rejected(self, store: TransitionStore) -> None:
        """Completing the same pending transition twice raises."""
        pending = store.begin(state=_state(0), action_index=0)
        pending.complete(next_state=_state(1), reward=0.0)
        with pytest.raises(TransitionError, match="already completed"):
            pending.complete(next_state=_state(2), reward=0.0)

    def test_complete_empty_next_state_rejected(self, store: TransitionStore) -> None:
        """Empty next_state is rejected."""
        pending = store.begin(state=_state(0), action_index=0)
        with pytest.raises(TransitionError, match="next_state must not be empty"):
            pending.complete(next_state=[], reward=0.0)

    def test_complete_nan_next_state_rejected(self, store: TransitionStore) -> None:
        """NaN in next_state is rejected."""
        pending = store.begin(state=_state(0), action_index=0)
        with pytest.raises(TransitionError, match="NaN"):
            pending.complete(next_state=[1.0, float("nan")], reward=0.0)

    def test_complete_inf_next_state_rejected(self, store: TransitionStore) -> None:
        """Inf in next_state is rejected."""
        pending = store.begin(state=_state(0), action_index=0)
        with pytest.raises(TransitionError, match="inf"):
            pending.complete(next_state=[1.0, float("inf")], reward=0.0)

    def test_complete_nan_reward_rejected(self, store: TransitionStore) -> None:
        """NaN reward is rejected."""
        pending = store.begin(state=_state(0), action_index=0)
        with pytest.raises(TransitionError, match="reward is NaN"):
            pending.complete(next_state=_state(1), reward=float("nan"))

    def test_complete_inf_reward_rejected(self, store: TransitionStore) -> None:
        """Inf reward is rejected."""
        pending = store.begin(state=_state(0), action_index=0)
        with pytest.raises(TransitionError, match="reward is NaN or inf"):
            pending.complete(next_state=_state(1), reward=float("inf"))

    def test_metadata_preserved(self, store: TransitionStore) -> None:
        """Metadata passed to complete() is preserved on the transition."""
        pending = store.begin(state=_state(0), action_index=0)
        transition = pending.complete(
            next_state=_state(1),
            reward=0.0,
            metadata={"scene": "test", "episode": 42},
        )
        assert transition.metadata["scene"] == "test"
        assert transition.metadata["episode"] == 42


# ========================================================================
# Timestamps and latency
# ========================================================================


class TestTimestamps:
    """Tests for timestamp monotonicity and latency measurement."""

    def test_timestamps_monotonic(self, store: TransitionStore) -> None:
        """completion >= start timestamp."""
        pending = store.begin(state=_state(0), action_index=0)
        time.sleep(0.001)  # small delay
        transition = pending.complete(next_state=_state(1), reward=0.0)
        assert transition.completion_timestamp_ns >= transition.start_timestamp_ns

    def test_latency_positive(self, store: TransitionStore) -> None:
        """Latency is non-negative."""
        pending = store.begin(state=_state(0), action_index=0)
        time.sleep(0.001)
        transition = pending.complete(next_state=_state(1), reward=0.0)
        assert transition.latency_ms >= 0.0

    def test_latency_measured_in_ms(self, store: TransitionStore) -> None:
        """Latency is measured in milliseconds."""
        pending = store.begin(state=_state(0), action_index=0)
        time.sleep(0.01)  # 10ms
        transition = pending.complete(next_state=_state(1), reward=0.0)
        # Should be at least ~10ms (with some slack)
        assert transition.latency_ms >= 5.0


# ========================================================================
# Callback
# ========================================================================


class TestCallback:
    """Tests for the on_transition_completed callback."""

    def test_callback_invoked_once(self, action_space: ActionSpace) -> None:
        """The callback is invoked exactly once per completed transition."""
        completed: list[Transition] = []
        store = TransitionStore(
            action_space=action_space,
            on_transition_completed=completed.append,
        )
        pending = store.begin(state=_state(0), action_index=0)
        transition = pending.complete(next_state=_state(1), reward=0.0)
        assert len(completed) == 1
        assert completed[0] is transition

    def test_callback_not_invoked_without_complete(self, action_space: ActionSpace) -> None:
        """The callback is not invoked when begin() is called without complete()."""
        completed: list[Transition] = []
        store = TransitionStore(
            action_space=action_space,
            on_transition_completed=completed.append,
        )
        store.begin(state=_state(0), action_index=0)
        assert len(completed) == 0


# ========================================================================
# Action identity: every transition must contain a real action
# ========================================================================


class TestActionIdentity:
    """Tests proving action identity comes from the ActionSpace."""

    def test_action_name_matches_space(
        self, store: TransitionStore, action_space: ActionSpace
    ) -> None:
        """The action_name in the transition matches the ActionSpace."""
        for i in range(action_space.size):
            pending = store.begin(state=_state(i), action_index=i)
            transition = pending.complete(next_state=_state(i + 1), reward=0.0)
            assert transition.action_name == action_space.get(i).name

    def test_action_vector_is_onehot(
        self, store: TransitionStore, action_space: ActionSpace
    ) -> None:
        """The action vector is the one-hot encoding from the ActionSpace."""
        pending = store.begin(state=_state(0), action_index=3)
        transition = pending.complete(next_state=_state(1), reward=0.0)
        assert len(transition.action_vector) == action_space.size
        assert transition.action_vector[3] == 1.0
        assert sum(transition.action_vector) == 1.0

    def test_observation_event_names_not_in_actions(
        self, store: TransitionStore, action_space: ActionSpace
    ) -> None:
        """Observation event types must never appear as action names."""
        observation_names = {
            "FaceDetected",
            "SpeechRecognized",
            "EmotionChanged",
            "IdleTimeout",
            "StateChanged",
            "ServoMoved",
        }
        action_names = {a.name for a in action_space}
        # No observation event name should be a valid action name
        assert observation_names.isdisjoint(action_names)


# ========================================================================
# Convenience: record() (begin + complete in one call)
# ========================================================================


class TestRecord:
    """Tests for the convenience record() method."""

    def test_record_produces_transition(self, store: TransitionStore) -> None:
        """record() produces a completed transition."""
        transition = store.record(
            state=_state(0),
            action_index=2,
            next_state=_state(1),
            reward=0.5,
        )
        assert transition.action_name == "look_center"
        assert transition.reward == 0.5
        assert store.pending_count == 0

    def test_record_validates_action(self, store: TransitionStore) -> None:
        """record() validates the action index."""
        with pytest.raises(TransitionError, match="out of range"):
            store.record(
                state=_state(0),
                action_index=999,
                next_state=_state(1),
                reward=0.0,
            )

    def test_record_validates_state(self, store: TransitionStore) -> None:
        """record() validates state vectors."""
        with pytest.raises(TransitionError, match="empty"):
            store.record(
                state=[],
                action_index=0,
                next_state=_state(1),
                reward=0.0,
            )


# ========================================================================
# to_experience: conversion to Experience for storage
# ========================================================================


class TestToExperience:
    """Tests for Transition.to_experience()."""

    def test_to_experience_has_action_vector(self, store: TransitionStore) -> None:
        """to_experience() uses the action one-hot vector as the action field."""
        pending = store.begin(state=_state(0), action_index=2)
        transition = pending.complete(next_state=_state(1), reward=0.5)
        exp = transition.to_experience()
        assert isinstance(exp, Experience)
        assert exp.action == transition.action_vector
        assert exp.reward == 0.5
        assert exp.state == transition.state
        assert exp.next_state == transition.next_state

    def test_to_experience_metadata_includes_execution_info(self, store: TransitionStore) -> None:
        """to_experience() preserves execution metadata."""
        pending = store.begin(
            state=_state(0),
            action_index=0,
            execution_id="exec-42",
            policy_version="v1.0",
        )
        transition = pending.complete(
            next_state=_state(1),
            reward=0.0,
            execution_success=False,
            execution_failure_reason="timeout",
            metadata={"scene": "test"},
        )
        exp = transition.to_experience()
        assert exp.metadata["execution_id"] == "exec-42"
        assert exp.metadata["action_index"] == 0
        assert exp.metadata["action_name"] == transition.action_name
        assert exp.metadata["execution_success"] is False
        assert exp.metadata["execution_failure_reason"] == "timeout"
        assert exp.metadata["policy_version"] == "v1.0"
        assert exp.metadata["scene"] == "test"
        assert exp.metadata["transition_id"] != ""
        assert exp.metadata["start_timestamp_ns"] > 0
        assert exp.metadata["completion_timestamp_ns"] > 0
        assert exp.metadata["latency_ms"] >= 0.0


# ========================================================================
# Definition of done: reconstruction test
# ========================================================================


class TestDefinitionOfDone:
    """Prove a stored transition can be reconstructed as the sequence:

    At T0: observation = X
    At T0: policy selected action = Y
    Robot executed Y
    At T1: observation = Z
    Reward = R
    """

    def test_reconstruct_transition_sequence(self, action_space: ActionSpace) -> None:
        """Full lifecycle: observe -> act -> observe outcome -> store."""
        completed: list[Transition] = []
        store = TransitionStore(
            action_space=action_space,
            on_transition_completed=completed.append,
        )

        # At T0: observation = X
        observation_t0 = _state(42)

        # At T0: policy selected action = Y (look_center, index 2)
        pending = store.begin(
            state=observation_t0,
            action_index=2,
            policy_version="deterministic",
        )

        # Robot executed Y (simulated)
        time.sleep(0.001)

        # At T1: observation = Z
        observation_t1 = _state(55)

        # Compute the reward
        reward = 0.75
        pending.complete(
            next_state=observation_t1,
            reward=reward,
            done=False,
        )

        # Reconstruct from storage
        assert len(completed) == 1
        stored = completed[0]

        # T0 observation
        assert stored.state == observation_t0, "State at T0 should match observation X"

        # Action Y
        assert stored.action_name == "look_center", "Action should be look_center (Y)"
        assert stored.action_index == 2
        assert stored.policy_version == "deterministic"

        # Execution happened
        assert stored.start_timestamp_ns < stored.completion_timestamp_ns
        assert stored.execution_success is True

        # T1 observation
        assert stored.next_state == observation_t1, "Next state at T1 should match observation Z"

        # Reward
        assert stored.reward == reward, f"Reward should be {reward}"

    def test_reconstruct_with_failure(self, action_space: ActionSpace) -> None:
        """Failed execution is reconstructable."""
        store = TransitionStore(action_space=action_space)

        pending = store.begin(state=_state(0), action_index=0)
        transition = pending.complete(
            next_state=_state(1),
            reward=-0.5,
            execution_success=False,
            execution_failure_reason="hardware error",
        )

        exp = transition.to_experience()
        assert exp.metadata["execution_success"] is False
        assert exp.metadata["execution_failure_reason"] == "hardware error"
        assert exp.reward == -0.5
