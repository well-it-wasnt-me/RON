"""Controlled online learning with separate robot and learning processes.

Allow production data to update the learning system — but only after
all prior safety gates are in place.

Key requirements:
* Separate processes: ``robot.service`` (perception, behavior, inference,
  hardware) and ``learning.service`` (replay, training, evaluation,
  candidate creation).
* Training must never starve the real-time control loop.
* Persist experiences, training runs, model versions, evaluation
  results, safety events, policy decisions.
* Warm the replay buffer from persistent storage after reboot.
* No unrestricted epsilon-greedy exploration on the physical robot —
  explore in simulation, offline replay, shadow mode, or constrained
  action subsets only.
* Monitor: model version, replay size, training rate, training loss,
  validation loss, reward, action distribution, safety rejections,
  fallback rate, sensor dropout, inference latency, model load failures.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from robot.learning.experience import ExperienceStore
from robot.logging import get_logger

_log = get_logger("learning.online")


# ---------------------------------------------------------------------------
# Online learning monitor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OnlineLearningMonitor:
    """Monitors the online learning system.

    Tracks all required online learning metrics:
    model version, replay size, training rate, training loss, validation
    loss, reward, action distribution, safety rejections, fallback rate,
    sensor dropout, inference latency, model load failures.
    """

    model_version: int = 0
    replay_size: int = 0
    training_rate_s: float = 0.0
    training_loss: float = float("inf")
    validation_loss: float = float("inf")
    total_reward: float = 0.0
    safety_rejections: int = 0
    fallback_count: int = 0
    sensor_dropout_count: int = 0
    inference_latency_ms: float = 0.0
    model_load_failures: int = 0

    # Action distribution (action_index → count)
    _action_counts: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    # Recent inference latencies for averaging
    _recent_latencies: deque[float] = field(
        default_factory=lambda: deque(maxlen=100), init=False, repr=False
    )
    # Training timestamps for rate calculation
    _training_timestamps: deque[float] = field(
        default_factory=lambda: deque(maxlen=100), init=False, repr=False
    )

    def record_action(self, action_index: int) -> None:
        """Record an executed action for distribution tracking."""
        self._action_counts[action_index] = self._action_counts.get(action_index, 0) + 1

    def record_safety_rejection(self) -> None:
        self.safety_rejections += 1

    def record_fallback(self) -> None:
        self.fallback_count += 1

    def record_sensor_dropout(self) -> None:
        self.sensor_dropout_count += 1

    def record_inference_latency(self, latency_ms: float) -> None:
        self._recent_latencies.append(latency_ms)
        if self._recent_latencies:
            self.inference_latency_ms = sum(self._recent_latencies) / len(self._recent_latencies)

    def record_model_load_failure(self) -> None:
        self.model_load_failures += 1

    def record_training(self, loss: float, val_loss: float = 0.0) -> None:
        """Record a training event."""
        self._training_timestamps.append(time.monotonic())
        self.training_loss = loss
        self.validation_loss = val_loss

        # Calculate training rate (trainings per second over last minute)
        now = time.monotonic()
        recent = [t for t in self._training_timestamps if now - t < 60.0]
        if len(recent) > 1:
            duration = recent[-1] - recent[0]
            if duration > 0:
                self.training_rate_s = len(recent) / duration

    def record_reward(self, reward: float) -> None:
        self.total_reward += reward

    @property
    def action_distribution(self) -> dict[int, float]:
        """Return action distribution as fractions."""
        total = sum(self._action_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self._action_counts.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "replay_size": self.replay_size,
            "training_rate_s": round(self.training_rate_s, 4),
            "training_loss": round(self.training_loss, 6),
            "validation_loss": round(self.validation_loss, 6),
            "total_reward": round(self.total_reward, 4),
            "action_distribution": {
                str(k): round(v, 4) for k, v in self.action_distribution.items()
            },
            "safety_rejections": self.safety_rejections,
            "fallback_count": self.fallback_count,
            "sensor_dropout_count": self.sensor_dropout_count,
            "inference_latency_ms": round(self.inference_latency_ms, 4),
            "model_load_failures": self.model_load_failures,
        }


# ---------------------------------------------------------------------------
# Exploration policy (constrained)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConstrainedExploration:
    """Constrained exploration for the physical robot.

    **Never** use unrestricted epsilon-greedy exploration on the physical
    robot.  Explore only in:
    - simulation
    - offline replay
    - shadow mode
    - constrained action subsets
    """

    allowed_action_indices: set[int] = field(default_factory=set)
    max_exploration_rate: float = 0.1
    min_exploration_rate: float = 0.01

    def is_action_allowed(self, action_index: int) -> bool:
        """Check if an action is in the allowed exploration subset."""
        if not self.allowed_action_indices:
            return True  # no restriction set
        return action_index in self.allowed_action_indices

    def clamp_rate(self, rate: float) -> float:
        """Clamp the exploration rate to safe bounds."""
        return max(self.min_exploration_rate, min(rate, self.max_exploration_rate))


# ---------------------------------------------------------------------------
# Persistent replay buffer warmer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReplayWarmer:
    """Warms the replay buffer from persistent storage after reboot.

    The robot process loads persisted experiences into the replay
    buffer so that training can resume immediately without waiting
    for ``min_new_experiences`` fresh events.

    Historical experiences are **not** counted as "new since last
    training" — only genuinely new events should trigger a training
    cycle.
    """

    store: ExperienceStore
    max_warm: int = 1000

    def warm(self, replay_buffer: object) -> int:
        """Load historical experiences into the replay buffer.

        Returns the number of experiences restored.
        """
        try:
            past = self.store.load_recent(limit=self.max_warm)
        except Exception:
            _log.exception("replay_warmer.load_failed")
            return 0

        restored = 0
        for exp in reversed(past):  # oldest first for ring buffer
            replay_buffer.add(exp)  # type: ignore[attr-defined]
            restored += 1

        _log.info("replay_warmer.warmed", count=restored)
        return restored


__all__ = [
    "ConstrainedExploration",
    "OnlineLearningMonitor",
    "ReplayWarmer",
]
