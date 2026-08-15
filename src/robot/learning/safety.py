"""Learning safety, evaluation, and rollback system.

This module implements safeguards that prevent a newly trained model from
degrading robot behavior:

* **Candidate evaluation** - compare candidate model against the current
  model on fixed evaluation scenarios before promotion.
* **Action validation** - verify that actions produced by the learning
  system are safe (within hardware limits, rate limits, etc.).
* **Rollback** - maintain previous checkpoints and support immediate
  rollback to the last known-good model.
* **Failure handling** - if learning crashes, the robot continues
  operating with the last valid model. Learning must be an enhancement,
  not a single point of failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot.learning.experience import Experience
from robot.learning.learning_service import CheckpointManager
from robot.learning.world_model import WorldModel
from robot.logging import get_logger

_log = get_logger("learning.safety")


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvaluationResult:
    """Result of evaluating a candidate model against the current model.

    Attributes
    ----------
    candidate_loss:
        Loss of the candidate model on the evaluation set.
    current_loss:
        Loss of the current model on the evaluation set.
    improvement_ratio:
        ``current_loss / candidate_loss`` - values > 1.0 mean the
        candidate is better.
    passed:
        Whether the candidate should be promoted.
    metric_details:
        Additional metrics (latency, resource usage, etc.).
    """

    candidate_loss: float = float("inf")
    current_loss: float = float("inf")
    improvement_ratio: float = 0.0
    passed: bool = False
    metric_details: dict[str, float] = field(default_factory=dict)

    @property
    def improvement_pct(self) -> float:
        """Percentage improvement (positive = better)."""
        if self.current_loss == 0:
            return 0.0
        return (1.0 - self.candidate_loss / self.current_loss) * 100.0


# ---------------------------------------------------------------------------
# Evaluation thresholds
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvaluationThresholds:
    """Configurable thresholds for candidate model evaluation.

    A candidate model is promoted only if ALL of these conditions pass.

    Attributes
    ----------
    min_improvement_ratio:
        Minimum improvement_ratio (current_loss / candidate_loss) to
        promote.  1.0 means "promote if equal or better"; 1.05 means
        "require at least 5% improvement".
    max_loss:
        Absolute maximum loss the candidate may have.  Prevents
        promoting a model that is terrible in absolute terms.
    max_prediction_latency_s:
        Maximum wall-clock time (seconds) for a single prediction.
        Prevents promoting a model that is too slow.
    max_prediction_std:
        Maximum standard deviation of predictions on the eval set.
        Prevents promoting a model that produces erratic outputs.
    """

    min_improvement_ratio: float = 1.0
    max_loss: float = float("inf")
    max_prediction_latency_s: float = 1.0
    max_prediction_std: float = float("inf")


# ---------------------------------------------------------------------------
# Action safety validator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ActionSafetyValidator:
    """Validates actions produced by the learning system.

    Ensures actions are within hardware limits, rate limits, and
    valid parameter ranges.  The learning system must **never**
    bypass these checks.
    """

    max_servo_angle: float = 180.0
    min_servo_angle: float = 0.0
    max_look_speed: float = 1.0
    max_blink_rate: float = 5.0  # blinks per second
    max_action_rate: float = 10.0  # actions per second
    allowed_action_names: set[str] = field(
        default_factory=lambda: {
            "look_left",
            "look_right",
            "look_center",
            "look_up",
            "look_down",
            "blink",
            "wink",
            "celebrate",
            "sleep",
            "look_around",
        }
    )

    _last_action_time: float = field(default=0.0, init=False, repr=False)
    _action_timestamps: list[float] = field(default_factory=list, init=False, repr=False)

    def validate_action(  # noqa: PLR0911, PLR0912
        self, action_name: str, params: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        """Validate an action and return (is_valid, reason).

        Parameters
        ----------
        action_name:
            The name of the action to validate.
        params:
            Optional action parameters.
        """
        # Check action name
        if action_name not in self.allowed_action_names:
            return False, f"unknown action: {action_name!r}"

        params = params or {}

        # Check rate limit
        now = time.monotonic()
        self._action_timestamps.append(now)
        # Keep only last second
        self._action_timestamps = [t for t in self._action_timestamps if now - t < 1.0]
        if len(self._action_timestamps) > self.max_action_rate:
            return False, f"rate limit exceeded: {len(self._action_timestamps)} actions/second"

        # Check servo parameters
        if "x" in params:
            x = params["x"]
            if not isinstance(x, (int, float)) or abs(x) > self.max_look_speed:
                return False, f"look x out of range: {x}"
        if "y" in params:
            y = params["y"]
            if not isinstance(y, (int, float)) or abs(y) > self.max_look_speed:
                return False, f"look y out of range: {y}"

        # Check servo angles
        for servo_param in ("angle",):
            if servo_param in params:
                angle = params[servo_param]
                if not isinstance(angle, (int, float)):
                    return False, f"invalid angle type: {type(angle)}"
                if angle < self.min_servo_angle or angle > self.max_servo_angle:
                    return False, f"angle out of range: {angle}"

        # Check blink speed
        if action_name in ("blink", "wink") and "speed" in params:
            speed = params["speed"]
            if not isinstance(speed, (int, float)) or speed > self.max_blink_rate or speed <= 0:
                return False, f"blink speed out of range: {speed}"

        # Check sleep duration
        if action_name == "sleep" and "duration_s" in params:
            duration = params["duration_s"]
            if not isinstance(duration, (int, float)) or duration <= 0 or duration > 300:
                return False, f"sleep duration out of range: {duration}s"

        return True, "ok"

    def validate_action_index(self, action_index: int, action_names: list[str]) -> tuple[bool, str]:
        """Validate an action index is within bounds."""
        if action_index < 0 or action_index >= len(action_names):
            return False, f"action index out of range: {action_index}"
        return self.validate_action(action_names[action_index])


# ---------------------------------------------------------------------------
# Model evaluator
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """Evaluates candidate models against the current model.

    Runs fixed evaluation scenarios and checks configurable thresholds
    before allowing a candidate model to be promoted.
    """

    def __init__(self, thresholds: EvaluationThresholds | None = None) -> None:
        self.thresholds = thresholds or EvaluationThresholds()

    def evaluate(  # noqa: PLR0912
        self,
        candidate: WorldModel,
        current: WorldModel,
        eval_experiences: list[Experience],
    ) -> EvaluationResult:
        """Evaluate a candidate model against the current model.

        Parameters
        ----------
        candidate:
            The model being considered for promotion.
        current:
            The currently active model.
        eval_experiences:
            Fixed evaluation set of experiences.

        Returns
        -------
        EvaluationResult
            Whether the candidate passed evaluation.
        """
        if not eval_experiences:
            return EvaluationResult(passed=False)

        # Measure prediction accuracy
        candidate_loss = candidate.evaluate(eval_experiences)
        current_loss = current.evaluate(eval_experiences)

        # Measure prediction latency
        if eval_experiences:
            state = np.array(eval_experiences[0].state, dtype=np.float64)
            action = np.array(eval_experiences[0].action, dtype=np.float64)
            # Pad action to model's action_size
            if len(action) < candidate.action_size:
                action = np.concatenate([action, np.zeros(candidate.action_size - len(action))])

            t0 = time.monotonic()
            for _ in range(10):
                candidate.predict(state.tolist(), action[: candidate.action_size].tolist())
            latency = (time.monotonic() - t0) / 10.0
        else:
            latency = 0.0

        # Measure prediction stability (std of predictions on eval set)
        predictions = []
        for exp in eval_experiences[:50]:
            s = np.array(exp.state, dtype=np.float64)
            a = np.array(exp.action, dtype=np.float64)
            if len(a) < candidate.action_size:
                a = np.concatenate([a, np.zeros(candidate.action_size - len(a))])
            pred = candidate.predict(s.tolist(), a[: candidate.action_size].tolist())
            predictions.append(pred)
        pred_std = float(np.std(predictions)) if predictions else 0.0

        # Calculate improvement ratio
        if current_loss > 0:
            improvement_ratio = current_loss / candidate_loss
        elif candidate_loss == 0:
            improvement_ratio = 1.0
        else:
            improvement_ratio = 0.0

        # Check all thresholds
        passed = True
        reasons: list[str] = []

        if improvement_ratio < self.thresholds.min_improvement_ratio:
            passed = False
            reasons.append(
                f"improvement_ratio {improvement_ratio:.4f} < {self.thresholds.min_improvement_ratio}"
            )

        if candidate_loss > self.thresholds.max_loss:
            passed = False
            reasons.append(f"candidate_loss {candidate_loss:.6f} > {self.thresholds.max_loss}")

        if latency > self.thresholds.max_prediction_latency_s:
            passed = False
            reasons.append(f"latency {latency:.4f}s > {self.thresholds.max_prediction_latency_s}s")

        if pred_std > self.thresholds.max_prediction_std:
            passed = False
            reasons.append(f"prediction_std {pred_std:.4f} > {self.thresholds.max_prediction_std}")

        result = EvaluationResult(
            candidate_loss=candidate_loss,
            current_loss=current_loss,
            improvement_ratio=improvement_ratio,
            passed=passed,
            metric_details={
                "latency_s": latency,
                "prediction_std": pred_std,
                "eval_size": len(eval_experiences),
            },
        )

        if not passed:
            _log.warning(
                "safety.evaluation_failed",
                candidate_loss=round(candidate_loss, 6),
                current_loss=round(current_loss, 6),
                improvement_ratio=round(improvement_ratio, 4),
                reasons=reasons,
            )

        return result


# ---------------------------------------------------------------------------
# Safety manager
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LearningSafetyManager:
    """Coordinates all safety mechanisms for the learning system.

    The safety manager ensures that:

    1.  Candidate models are evaluated before promotion.
    2.  Actions are validated before execution.
    3.  Checkpoints are maintained for rollback.
    4.  Failures are handled gracefully.
    """

    evaluator: ModelEvaluator = field(default_factory=ModelEvaluator)
    action_validator: ActionSafetyValidator = field(default_factory=ActionSafetyValidator)
    checkpoint_manager: CheckpointManager | None = None

    def evaluate_candidate(
        self,
        candidate: WorldModel,
        current: WorldModel,
        eval_experiences: list[Experience],
    ) -> EvaluationResult:
        """Evaluate a candidate model for promotion."""
        return self.evaluator.evaluate(candidate, current, eval_experiences)

    def validate_action(
        self, action_name: str, params: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        """Validate an action before execution."""
        return self.action_validator.validate_action(action_name, params)

    def should_promote(self, evaluation: EvaluationResult) -> bool:
        """Decide whether to promote based on evaluation results."""
        return evaluation.passed


__all__ = [
    "ActionSafetyValidator",
    "EvaluationResult",
    "EvaluationThresholds",
    "LearningSafetyManager",
    "ModelEvaluator",
]
