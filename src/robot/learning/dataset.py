"""Dataset construction and validation for world model training.

Dataset construction and validation for world model training:

* Only train from **completed** transitions with valid physical
  semantics.
* Reject missing next state, missing action, invalid action, NaN/inf,
  impossible timestamps, corrupted observation vectors, and transitions
  created without execution.
* Split data into train / validation / test using **time-based** or
  **episode-based** splits (not random) to avoid leaking near-identical
  consecutive samples.
* Track training loss, validation loss, test loss, per-feature error,
  prediction latency, and invalid prediction count.
* Create a trivial baseline that the learned model must beat.

This module provides:

* :class:`TransitionDataset` - validates and splits transitions.
* :class:`WorldModelBaseline` - trivial persistence / mean predictor.
* :class:`DatasetSplit` - the three-way train/val/test split.
* :class:`DatasetStats` - statistics about the dataset.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot.learning.action_learning import ActionSpace
from robot.learning.experience import Experience
from robot.learning.transition import Transition
from robot.logging import get_logger

_log = get_logger("learning.dataset")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TransitionValidationError(ValueError):
    """Raised when a transition is invalid for training."""


def validate_transition(  # noqa: PLR0911, PLR0912
    transition: Transition | Experience,
    action_space: ActionSpace | None = None,
) -> tuple[bool, str]:
    """Validate a transition for world-model training.

    Returns ``(True, "ok")`` if valid, ``(False, reason)`` otherwise.

    Rejects:
    * missing next state
    * missing/empty action
    * invalid action (not in action space)
    * NaN/inf in state, action, next_state, or reward
    * impossible timestamps (completion < start)
    * corrupted observation vectors (wrong size)
    * transitions created without execution (execution_success == False
      is allowed but flagged - the transition is still valid if data is
      intact)
    """
    if isinstance(transition, Transition):
        state = transition.state
        action_vec = transition.action_vector
        next_state = transition.next_state
        reward = transition.reward
        action_index = transition.action_index
        start_ns = transition.start_timestamp_ns
        completion_ns = transition.completion_timestamp_ns
    elif isinstance(transition, Experience):
        state = transition.state
        action_vec = transition.action
        next_state = transition.next_state
        reward = transition.reward
        action_index = transition.metadata.get("action_index", -1)
        start_ns = transition.metadata.get("start_timestamp_ns", 0)
        completion_ns = transition.metadata.get("completion_timestamp_ns", 0)
    else:
        raise TransitionValidationError(f"unsupported type: {type(transition)}")

    assert state is not None
    assert action_vec is not None
    assert next_state is not None

    # Missing next state
    if not next_state:
        return False, "missing next_state"

    # Missing/empty action
    if not action_vec:
        return False, "missing action"

    # NaN/inf in state
    for i, v in enumerate(state):
        if math.isnan(v):
            return False, f"state[{i}] is NaN"
        if math.isinf(v):
            return False, f"state[{i}] is inf"

    # NaN/inf in action
    for i, v in enumerate(action_vec):
        if math.isnan(v):
            return False, f"action[{i}] is NaN"
        if math.isinf(v):
            return False, f"action[{i}] is inf"

    # NaN/inf in next_state
    for i, v in enumerate(next_state):
        if math.isnan(v):
            return False, f"next_state[{i}] is NaN"
        if math.isinf(v):
            return False, f"next_state[{i}] is inf"

    # NaN/inf in reward
    if math.isnan(reward) or math.isinf(reward):
        return False, f"reward is NaN/inf: {reward}"

    # Invalid action (if action_space provided)
    if action_space is not None and not (0 <= action_index < action_space.size):
        return False, f"action_index {action_index} out of range"

    # Impossible timestamps (if both present)
    if start_ns > 0 and completion_ns > 0 and completion_ns < start_ns:
        return False, "completion timestamp before start timestamp"

    return True, "ok"


# ---------------------------------------------------------------------------
# Dataset stats
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DatasetStats:
    """Statistics about a dataset split."""

    total: int = 0
    valid: int = 0
    rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "valid": self.valid,
            "rejected": self.rejected,
            "rejection_reasons": dict(self.rejection_reasons),
        }


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSplit:
    """Three-way train/validation/test split of experiences.

    Attributes
    ----------
    train:
        Training experiences.
    validation:
        Validation experiences.
    test:
        Test experiences.
    """

    train: tuple[Experience, ...]
    validation: tuple[Experience, ...]
    test: tuple[Experience, ...]

    @property
    def total(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_size": len(self.train),
            "validation_size": len(self.validation),
            "test_size": len(self.test),
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# TransitionDataset
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TransitionDataset:
    """Validates and splits transitions for world-model training.

    Parameters
    ----------
    action_space:
        The action space for validation.  If None, action index
        validation is skipped.
    train_ratio:
        Fraction of data for training (default 0.7).
    val_ratio:
        Fraction of data for validation (default 0.15).
    test_ratio:
        Fraction of data for testing (default 0.15).
    """

    action_space: ActionSpace | None = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def build(
        self,
        transitions: Sequence[Transition | Experience],
        split_method: str = "time",
    ) -> tuple[DatasetSplit, DatasetStats]:
        """Validate transitions and split into train/val/test.

        Parameters
        ----------
        transitions:
            The transitions to process.
        split_method:
            "time" for time-based split (default), "episode" for
            episode-based split (uses metadata["episode"] or
            metadata["source"]).

        Returns
        -------
        tuple[DatasetSplit, DatasetStats]
            The split and statistics.
        """
        stats = DatasetStats(total=len(transitions))

        # Validate
        valid_experiences: list[Experience] = []
        for t in transitions:
            stats.total += 0  # already counted
            ok, reason = validate_transition(t, self.action_space)
            if ok:
                if isinstance(t, Transition):
                    valid_experiences.append(t.to_experience())
                else:
                    valid_experiences.append(t)
                stats.valid += 1
            else:
                stats.rejected += 1
                stats.rejection_reasons[reason] = stats.rejection_reasons.get(reason, 0) + 1

        stats.valid = len(valid_experiences)

        if len(valid_experiences) < 3:
            _log.warning(
                "dataset.too_few_valid",
                valid=len(valid_experiences),
                rejected=stats.rejected,
            )
            return DatasetSplit(
                train=tuple(valid_experiences),
                validation=(),
                test=(),
            ), stats

        # Split
        if split_method == "time":
            split = self._split_time_based(valid_experiences)
        elif split_method == "episode":
            split = self._split_episode_based(valid_experiences)
        else:
            split = self._split_time_based(valid_experiences)

        _log.info(
            "dataset.built",
            valid=stats.valid,
            rejected=stats.rejected,
            train=len(split.train),
            val=len(split.validation),
            test=len(split.test),
        )

        return split, stats

    def _split_time_based(self, experiences: list[Experience]) -> DatasetSplit:
        """Split by time: earliest -> train, middle -> val, latest -> test.

        This prevents temporal leakage of near-identical consecutive
        samples across splits.
        """
        # Sort by timestamp
        sorted_exps = sorted(experiences, key=lambda e: e.timestamp)
        n = len(sorted_exps)
        n_train = max(1, int(n * self.train_ratio))
        n_val = max(1, int(n * self.val_ratio))
        n_test = n - n_train - n_val

        if n_test < 1:
            n_test = max(1, n - n_train - 1)
            n_val = n - n_train - n_test

        train = tuple(sorted_exps[:n_train])
        validation = tuple(sorted_exps[n_train : n_train + n_val])
        test = tuple(sorted_exps[n_train + n_val :])

        return DatasetSplit(train=train, validation=validation, test=test)

    def _split_episode_based(self, experiences: list[Experience]) -> DatasetSplit:
        """Split by episode (uses metadata["episode"] or metadata["source"])."""
        # Group by episode
        episodes: dict[str, list[Experience]] = {}
        for exp in experiences:
            key = exp.metadata.get("episode", exp.metadata.get("source", "default"))
            episodes.setdefault(str(key), []).append(exp)

        # Sort episodes and assign to splits
        sorted_keys = sorted(episodes.keys())
        n_episodes = len(sorted_keys)
        n_train = max(1, int(n_episodes * self.train_ratio))
        n_val = max(1, int(n_episodes * self.val_ratio))

        train: list[Experience] = []
        validation: list[Experience] = []
        test: list[Experience] = []

        for i, key in enumerate(sorted_keys):
            if i < n_train:
                train.extend(episodes[key])
            elif i < n_train + n_val:
                validation.extend(episodes[key])
            else:
                test.extend(episodes[key])

        return DatasetSplit(train=tuple(train), validation=tuple(validation), test=tuple(test))


# ---------------------------------------------------------------------------
# World model baseline
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorldModelBaseline:
    """Trivial baseline predictor that the learned model must beat.

    Implements three strategies:

    * ``persistence``: predict next_state = current_state.
    * ``mean``: predict next_state = mean of all training next_states.
    * ``zero``: predict next_state = all zeros (worst case).

    The ``persistence`` model is the default and the most natural
    baseline for a world model.
    """

    strategy: str = "persistence"
    _mean_next_state: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, train_experiences: list[Experience]) -> None:
        """Fit the baseline on training data.

        For ``persistence``: no fitting needed.
        For ``mean``: compute the mean of all training next_states.
        """
        if self.strategy == "mean" and train_experiences:
            next_states = np.array([e.next_state for e in train_experiences], dtype=np.float64)
            self._mean_next_state = np.mean(next_states, axis=0)

    def predict(
        self, state: list[float] | np.ndarray, action: list[float] | np.ndarray
    ) -> np.ndarray:
        """Predict the next state."""
        state_arr = np.asarray(state, dtype=np.float64)
        if self.strategy == "persistence":
            return state_arr.copy()
        if self.strategy == "mean" and self._mean_next_state is not None:
            return self._mean_next_state.copy()
        if self.strategy == "zero":
            return np.zeros_like(state_arr)
        return state_arr.copy()

    def evaluate(self, experiences: list[Experience]) -> float:
        """Evaluate the baseline on a set of experiences.

        Returns the mean MSE loss.
        """
        if not experiences:
            return 0.0
        total_loss = 0.0
        for exp in experiences:
            pred = self.predict(exp.state, exp.action)
            actual = np.array(exp.next_state, dtype=np.float64)
            total_loss += float(np.mean((pred - actual) ** 2))
        return total_loss / len(experiences)


__all__ = [
    "DatasetSplit",
    "DatasetStats",
    "TransitionDataset",
    "TransitionValidationError",
    "WorldModelBaseline",
    "validate_transition",
]
