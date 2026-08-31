"""Shadow-mode policy: run the learned policy alongside the deterministic controller.

Until shadow mode is proven safe, the learned policy must not control
the robot.

Modes
-----

* ``off`` - no learned inference at all.
* ``shadow`` - the model predicts an action but the deterministic
  controller executes the real action.  The learned action is logged
  for comparison.
* ``assist`` - the model may suggest actions, but deterministic logic
  can reject them.
* ``active`` - only explicitly approved actions can be controlled by
  the learned policy.  **Not enabled in this phase.**

In shadow mode, for every decision we log:

    timestamp, observation ID, current behavior action,
    model action, model scores/Q-values, model version,
    safety result, inference latency.

We measure: policy agreement, predicted reward, disagreement cases,
unsafe proposals, latency, confidence/Q-value margins.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from robot.learning.action_learning import ActionSpace
from robot.logging import get_logger

_log = get_logger("learning.shadow_policy")


# ---------------------------------------------------------------------------
# Policy mode
# ---------------------------------------------------------------------------


class PolicyMode(str, Enum):
    """Control modes for the learned policy.

    * ``off`` - no learned inference.
    * ``shadow`` - model predicts, deterministic controller executes.
    * ``assist`` - model may suggest, deterministic logic can reject.
    * ``active`` - only approved actions controlled by learned policy.
    """

    OFF = "off"
    SHADOW = "shadow"
    ASSIST = "assist"
    ACTIVE = "active"


# ---------------------------------------------------------------------------
# Shadow log entry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ShadowLogEntry:
    """A single decision logged in shadow mode.

    Attributes
    ----------
    timestamp_ns:
        Monotonic nanosecond timestamp.
    observation_id:
        Identifier for the observation being evaluated.
    deterministic_action:
        Action index selected by the deterministic controller.
    model_action:
        Action index selected by the learned model.
    model_scores:
        Q-values or scores for all actions.
    model_version:
        Version string of the model.
    safety_result:
        "ok" if the model action passed safety, "rejected" otherwise.
    inference_latency_ms:
        Time taken for model inference.
    agreement:
        Whether model and deterministic controller agreed.
    """

    timestamp_ns: int
    observation_id: str
    deterministic_action: int
    model_action: int
    model_scores: list[float]
    model_version: str
    safety_result: str
    inference_latency_ms: float
    agreement: bool


# ---------------------------------------------------------------------------
# Shadow metrics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ShadowMetrics:
    """Aggregate metrics from shadow-mode operation.

    Attributes
    ----------
    total_decisions:
        Total number of shadow decisions logged.
    agreement_count:
        Number of times model and deterministic controller agreed.
    disagreement_count:
        Number of times they disagreed.
    unsafe_proposals:
        Number of model actions that were unsafe.
    predicted_reward:
        Sum of predicted rewards from the model.
    total_latency_ms:
        Total inference latency.
    max_latency_ms:
        Maximum inference latency.
    model_version:
        Version of the model being shadowed.
    """

    total_decisions: int = 0
    agreement_count: int = 0
    disagreement_count: int = 0
    unsafe_proposals: int = 0
    predicted_reward: float = 0.0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    model_version: str = ""

    @property
    def agreement_rate(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return self.agreement_count / self.total_decisions

    @property
    def avg_latency_ms(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return self.total_latency_ms / self.total_decisions

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "agreement_count": self.agreement_count,
            "disagreement_count": self.disagreement_count,
            "unsafe_proposals": self.unsafe_proposals,
            "agreement_rate": round(self.agreement_rate, 4),
            "predicted_reward": round(self.predicted_reward, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 4),
            "max_latency_ms": round(self.max_latency_ms, 4),
            "model_version": self.model_version,
        }


# ---------------------------------------------------------------------------
# Shadow policy controller
# ---------------------------------------------------------------------------


DeterministicPolicy = Callable[[np.ndarray], int]
ModelPolicy = Callable[[np.ndarray], tuple[int, list[float]]]
SafetyChecker = Callable[[int, np.ndarray, ActionSpace], tuple[bool, str]]


@dataclass(slots=True)
class ShadowPolicyController:
    """Runs the learned policy in shadow mode alongside the deterministic controller.

    In shadow mode, the model predicts an action but the deterministic
    controller executes the real action.  The model action is logged
    for comparison but **never** executed on hardware.

    Parameters
    ----------
    action_space:
        The action space.
    deterministic_policy:
        Function that takes a state vector and returns an action index.
        This is the controller that actually executes.
    model_policy:
        Function that takes a state vector and returns
        ``(action_index, scores)``.  This is the learned policy.
    safety_checker:
        Optional function that validates the model's proposed action.
    model_version:
        Version string of the model being shadowed.
    mode:
        Current policy mode (default: shadow).
    max_log_entries:
        Maximum number of log entries to keep.
    """

    action_space: ActionSpace
    deterministic_policy: DeterministicPolicy
    model_policy: ModelPolicy | None = None
    safety_checker: SafetyChecker | None = None
    model_version: str = "unknown"
    mode: PolicyMode = PolicyMode.SHADOW
    max_log_entries: int = 10000

    _log_entries: list[ShadowLogEntry] = field(default_factory=list, init=False, repr=False)
    _metrics: ShadowMetrics = field(default_factory=ShadowMetrics, init=False, repr=False)

    def __post_init__(self) -> None:
        self._metrics.model_version = self.model_version

    @property
    def metrics(self) -> ShadowMetrics:
        """Return a snapshot of the aggregate metrics."""
        return ShadowMetrics(
            total_decisions=self._metrics.total_decisions,
            agreement_count=self._metrics.agreement_count,
            disagreement_count=self._metrics.disagreement_count,
            unsafe_proposals=self._metrics.unsafe_proposals,
            predicted_reward=self._metrics.predicted_reward,
            total_latency_ms=self._metrics.total_latency_ms,
            max_latency_ms=self._metrics.max_latency_ms,
            model_version=self._metrics.model_version,
        )

    @property
    def log_entries(self) -> list[ShadowLogEntry]:
        return list(self._log_entries)

    def decide(self, state: np.ndarray, observation_id: str = "") -> int:
        """Run a shadow decision cycle.

        In shadow mode:
        1. The deterministic controller selects the real action.
        2. The model predicts an action (for logging only).
        3. The model action is checked for safety.
        4. Everything is logged.
        5. The deterministic action is returned (NOT the model action).

        Returns the deterministic action index.
        """
        if self.mode == PolicyMode.OFF:
            return self.deterministic_policy(state)

        # Deterministic controller always selects the real action
        deterministic_action = self.deterministic_policy(state)

        # Model predicts (for logging/comparison only)
        model_action = deterministic_action
        model_scores: list[float] = []
        safety_result = "ok"
        inference_latency = 0.0

        if self.model_policy is not None and self.mode in (PolicyMode.SHADOW, PolicyMode.ASSIST):
            t0 = time.monotonic()
            try:
                model_action, model_scores = self.model_policy(state)
            except Exception:
                _log.exception("shadow_policy.model_error")
                model_action = -1
                model_scores = []
                safety_result = "model_error"
            inference_latency = (time.monotonic() - t0) * 1000

            # Safety check the model's proposed action
            if self.safety_checker is not None and model_action >= 0:
                ok, reason = self.safety_checker(model_action, state, self.action_space)
                safety_result = "ok" if ok else f"rejected: {reason}"
            elif model_action < 0:
                safety_result = "model_error"

        # Log the decision
        agreement = model_action == deterministic_action
        entry = ShadowLogEntry(
            timestamp_ns=time.monotonic_ns(),
            observation_id=observation_id,
            deterministic_action=deterministic_action,
            model_action=model_action,
            model_scores=model_scores,
            model_version=self.model_version,
            safety_result=safety_result,
            inference_latency_ms=inference_latency,
            agreement=agreement,
        )
        self._log_entries.append(entry)
        if len(self._log_entries) > self.max_log_entries:
            self._log_entries.pop(0)

        # Update metrics
        self._metrics.total_decisions += 1
        if agreement:
            self._metrics.agreement_count += 1
        else:
            self._metrics.disagreement_count += 1
        if "rejected" in safety_result or "error" in safety_result:
            self._metrics.unsafe_proposals += 1
        self._metrics.total_latency_ms += inference_latency
        self._metrics.max_latency_ms = max(self._metrics.max_latency_ms, inference_latency)

        # In shadow mode, always return the deterministic action.
        # In assist mode, return the model action when it passes safety.
        if self.mode == PolicyMode.ASSIST and safety_result == "ok" and model_action >= 0:
            return model_action
        return deterministic_action

    def set_mode(self, mode: PolicyMode) -> None:
        """Change the policy mode."""
        if mode == PolicyMode.ACTIVE:
            raise NotImplementedError(
                "PolicyMode.ACTIVE is not yet implemented - use SHADOW or ASSIST."
            )
        self.mode = mode
        _log.info("shadow_policy.mode_changed", mode=mode.value)

    def reset_metrics(self) -> None:
        """Reset the metrics and clear logs."""
        self._log_entries.clear()
        self._metrics = ShadowMetrics(model_version=self.model_version)


__all__ = [
    "DeterministicPolicy",
    "ModelPolicy",
    "PolicyMode",
    "SafetyChecker",
    "ShadowLogEntry",
    "ShadowMetrics",
    "ShadowPolicyController",
]
