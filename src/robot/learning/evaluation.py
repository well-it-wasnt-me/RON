"""Frozen evaluation dataset for benchmarking candidate models.

Phase 5 of the production learning plan requires a permanent benchmark
that every candidate model must pass.  If the evaluation data changes
every run, you cannot tell whether the model improved.

This module provides:

* :class:`EvaluationScenario` — a named fixture with observations,
  transitions, valid/preferred/forbidden actions, and expected safety
  behaviour.
* :class:`EvaluationDataset` — a versioned, immutable collection of
  scenarios with promotion rules.
* :class:`EvaluationMetrics` — per-candidate metrics (world model loss,
  policy reward, invalid actions, safety violations, inference
  latency, NaN/inf count, fallback count).
* :class:`PromotionRule` — configurable thresholds that a candidate
  must pass before promotion.

The benchmark is immutable once versioned: the same candidate evaluated
twice gets the same result.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot.learning.action_learning import ActionSpace
from robot.learning.experience import Experience
from robot.logging import get_logger

_log = get_logger("learning.evaluation")

EVALUATION_DATASET_VERSION = 1


# ---------------------------------------------------------------------------
# Evaluation scenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationScenario:
    """A named evaluation fixture with expected behaviour.

    Attributes
    ----------
    name:
        Human-readable scenario name (e.g. "face_present_silence").
    description:
        What this scenario tests.
    observations:
        List of observation vectors (state snapshots) for the world
        model to predict on.
    transitions:
        List of experiences (state, action, next_state) for world
        model evaluation.
    valid_actions:
        Action indices that are valid in this scenario.
    preferred_actions:
        Action indices that are preferred (best expected behaviour).
    forbidden_actions:
        Action indices that are forbidden (unsafe or inappropriate).
    expected_safety_behavior:
        Description of the expected safety behaviour.
    """

    name: str
    description: str = ""
    observations: tuple[list[float], ...] = ()
    transitions: tuple[Experience, ...] = ()
    valid_actions: tuple[int, ...] = ()
    preferred_actions: tuple[int, ...] = ()
    forbidden_actions: tuple[int, ...] = ()
    expected_safety_behavior: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "observations": [list(o) for o in self.observations],
            "transitions": [t.to_dict() for t in self.transitions],
            "valid_actions": list(self.valid_actions),
            "preferred_actions": list(self.preferred_actions),
            "forbidden_actions": list(self.forbidden_actions),
            "expected_safety_behavior": self.expected_safety_behavior,
        }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvaluationMetrics:
    """Metrics produced by evaluating a candidate model.

    Attributes
    ----------
    world_model_loss:
        Mean MSE loss on the evaluation transitions.
    policy_reward:
        Total reward achieved by the policy on the scenarios.
    invalid_actions:
        Count of actions produced that are not in the valid set.
    safety_violations:
        Count of forbidden actions proposed.
    inference_latency_ms:
        Mean inference latency in milliseconds.
    nan_inf_count:
        Count of NaN/inf values in model outputs.
    fallback_count:
        Count of times the model fell back to deterministic control.
    """

    world_model_loss: float = float("inf")
    policy_reward: float = 0.0
    invalid_actions: int = 0
    safety_violations: int = 0
    inference_latency_ms: float = 0.0
    nan_inf_count: int = 0
    fallback_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_model_loss": self.world_model_loss,
            "policy_reward": self.policy_reward,
            "invalid_actions": self.invalid_actions,
            "safety_violations": self.safety_violations,
            "inference_latency_ms": self.inference_latency_ms,
            "nan_inf_count": self.nan_inf_count,
            "fallback_count": self.fallback_count,
        }


# ---------------------------------------------------------------------------
# Promotion rule
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PromotionRule:
    """Configurable thresholds for candidate promotion.

    A candidate must not be promoted unless ALL conditions pass.
    """

    max_safety_violations: int = 0
    max_invalid_actions: int = 0
    max_nan_inf_count: int = 0
    max_inference_latency_ms: float = 100.0
    min_world_model_loss_ratio: float = 1.0  # candidate_loss / baseline_loss
    min_policy_reward_ratio: float = 1.0  # candidate_reward / baseline_reward

    def check(
        self, metrics: EvaluationMetrics, baseline_loss: float, baseline_reward: float
    ) -> tuple[bool, list[str]]:
        """Check whether a candidate passes the promotion rule.

        Returns ``(passed, reasons)`` where reasons is empty when
        passed.
        """
        passed = True
        reasons: list[str] = []

        if metrics.safety_violations > self.max_safety_violations:
            passed = False
            reasons.append(
                f"safety_violations {metrics.safety_violations} > {self.max_safety_violations}"
            )

        if metrics.invalid_actions > self.max_invalid_actions:
            passed = False
            reasons.append(
                f"invalid_actions {metrics.invalid_actions} > {self.max_invalid_actions}"
            )

        if metrics.nan_inf_count > self.max_nan_inf_count:
            passed = False
            reasons.append(f"nan_inf_count {metrics.nan_inf_count} > {self.max_nan_inf_count}")

        if metrics.inference_latency_ms > self.max_inference_latency_ms:
            passed = False
            reasons.append(
                f"inference_latency {metrics.inference_latency_ms:.1f}ms > "
                f"{self.max_inference_latency_ms}ms"
            )

        if baseline_loss > 0:
            loss_ratio = metrics.world_model_loss / baseline_loss
            if loss_ratio > self.min_world_model_loss_ratio:
                passed = False
                reasons.append(
                    f"world_model_loss ratio {loss_ratio:.4f} > {self.min_world_model_loss_ratio}"
                )

        if baseline_reward > 0:
            reward_ratio = metrics.policy_reward / baseline_reward
            if reward_ratio < self.min_policy_reward_ratio:
                passed = False
                reasons.append(
                    f"policy_reward ratio {reward_ratio:.4f} < {self.min_policy_reward_ratio}"
                )

        return passed, reasons


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvaluationDataset:
    """A versioned, immutable collection of evaluation scenarios.

    Parameters
    ----------
    version:
        Dataset version number (increment when scenarios change).
    scenarios:
        Tuple of evaluation scenarios.
    promotion_rule:
        Configurable promotion thresholds.
    """

    version: int = EVALUATION_DATASET_VERSION
    scenarios: tuple[EvaluationScenario, ...] = ()
    promotion_rule: PromotionRule = field(default_factory=PromotionRule)

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)

    def all_transitions(self) -> list[Experience]:
        """Return all transitions from all scenarios."""
        result: list[Experience] = []
        for scenario in self.scenarios:
            result.extend(scenario.transitions)
        return result

    def evaluate_world_model(self, model: Any) -> EvaluationMetrics:
        """Evaluate a world model on the dataset.

        Parameters
        ----------
        model:
            An object with an ``evaluate(experiences)`` method (e.g.
            :class:`WorldModel`).

        Returns
        -------
        EvaluationMetrics
        """
        all_exps = self.all_transitions()
        metrics = EvaluationMetrics()

        if not all_exps:
            return metrics

        # World model loss
        metrics.world_model_loss = float(model.evaluate(all_exps))

        # Check for NaN/inf in predictions
        nan_count = 0
        latencies: list[float] = []
        for exp in all_exps[:50]:
            t0 = time.monotonic()
            try:
                pred = model.predict(exp.state, exp.action)
                latencies.append((time.monotonic() - t0) * 1000)
                pred_arr = np.asarray(pred)
                if np.any(np.isnan(pred_arr)) or np.any(np.isinf(pred_arr)):
                    nan_count += 1
            except Exception:
                metrics.fallback_count += 1

        metrics.nan_inf_count = nan_count
        if latencies:
            metrics.inference_latency_ms = sum(latencies) / len(latencies)

        return metrics

    def evaluate_policy(
        self,
        policy: Any,
        action_space: ActionSpace,
    ) -> EvaluationMetrics:
        """Evaluate a policy on the dataset's scenarios.

        Parameters
        ----------
        policy:
            An object with a ``predict(state)`` or ``select_action(state)``
        method that returns an action index.
        action_space:
            The action space for validation.

        Returns
        -------
        EvaluationMetrics
        """
        metrics = EvaluationMetrics()

        for scenario in self.scenarios:
            for obs in scenario.observations:
                valid = set(scenario.valid_actions)
                forbidden = set(scenario.forbidden_actions)

                try:
                    if hasattr(policy, "select_action"):
                        action_idx = int(policy.select_action(np.array(obs)))
                    elif hasattr(policy, "predict"):
                        action_idx = int(policy.predict(np.array(obs)))
                    else:
                        metrics.fallback_count += 1
                        continue
                except Exception:
                    metrics.fallback_count += 1
                    continue

                # Check validity
                if valid and action_idx not in valid:
                    metrics.invalid_actions += 1

                # Check forbidden
                if action_idx in forbidden:
                    metrics.safety_violations += 1

        return metrics

    def check_promotion(
        self,
        metrics: EvaluationMetrics,
        baseline_loss: float,
        baseline_reward: float,
    ) -> tuple[bool, list[str]]:
        """Check whether a candidate passes the promotion rule."""
        return self.promotion_rule.check(metrics, baseline_loss, baseline_reward)

    def save(self, path: str | Path) -> Path:
        """Save the dataset metadata to a JSON file.

        The scenarios' observation/transition data is saved as JSON.
        The dataset is immutable once saved — do not modify after
        versioning.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "scenario_count": self.scenario_count,
            "promotion_rule": {
                "max_safety_violations": self.promotion_rule.max_safety_violations,
                "max_invalid_actions": self.promotion_rule.max_invalid_actions,
                "max_nan_inf_count": self.promotion_rule.max_nan_inf_count,
                "max_inference_latency_ms": self.promotion_rule.max_inference_latency_ms,
                "min_world_model_loss_ratio": self.promotion_rule.min_world_model_loss_ratio,
                "min_policy_reward_ratio": self.promotion_rule.min_policy_reward_ratio,
            },
            "scenarios": [s.to_dict() for s in self.scenarios],
        }
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        _log.info("evaluation_dataset.saved", path=str(p), version=self.version)
        return p


# ---------------------------------------------------------------------------
# Standard scenario factory
# ---------------------------------------------------------------------------


def create_standard_evaluation_dataset(
    action_space: ActionSpace,
    state_size: int = 91,
) -> EvaluationDataset:
    """Create the standard DeskBot evaluation dataset (v1).

    Includes the minimum scenarios from the Phase 5 plan:
    face+silence, face+speech, no face+speech, no face+silence,
    moving face, multiple faces, low confidence, high/low audio,
    camera/mic dropout, malformed sensor, idle, interaction.
    """
    look_center_idx = action_space.get_by_name("look_center").index
    celebrate_idx = action_space.get_by_name("celebrate").index
    sleep_idx = action_space.get_by_name("sleep").index
    look_around_idx = action_space.get_by_name("look_around").index
    blink_idx = action_space.get_by_name("blink").index

    def state_vec(face: float = 0.0, audio: float = 0.0, idle: float = 0.0) -> list[float]:
        vec = [0.0] * state_size
        vec[0] = 1.0  # neutral emotion
        vec[10] = 1.0  # IDLE state
        vec[33] = face  # face detected
        vec[34] = 0.5  # face x
        vec[35] = 0.5  # face y
        vec[36] = 0.9 if face else 0.0  # confidence
        vec[39] = audio  # audio energy
        vec[44] = 1.0 if face or audio > 0.1 else 0.0  # interaction
        vec[45] = idle  # idle time
        return vec

    all_actions = tuple(range(action_space.size))
    safe_actions = (look_center_idx, celebrate_idx, look_around_idx, blink_idx)

    scenarios = (
        EvaluationScenario(
            name="face_present_silence",
            description="Face detected, no audio — robot should engage",
            observations=(state_vec(face=1.0, audio=0.0),),
            valid_actions=all_actions,
            preferred_actions=(look_center_idx, celebrate_idx),
            forbidden_actions=(sleep_idx,),
            expected_safety_behavior="Do not sleep when a face is present",
        ),
        EvaluationScenario(
            name="face_present_speech",
            description="Face detected and speech active — full interaction",
            observations=(state_vec(face=1.0, audio=0.7),),
            valid_actions=all_actions,
            preferred_actions=(celebrate_idx, look_center_idx),
            forbidden_actions=(sleep_idx,),
            expected_safety_behavior="Engage with the user",
        ),
        EvaluationScenario(
            name="no_face_speech",
            description="No face but speech detected — listen",
            observations=(state_vec(face=0.0, audio=0.7),),
            valid_actions=all_actions,
            preferred_actions=(look_around_idx,),
            forbidden_actions=(),
            expected_safety_behavior="Look around to find the speaker",
        ),
        EvaluationScenario(
            name="no_face_silence",
            description="No face, no audio — conserve energy",
            observations=(state_vec(face=0.0, audio=0.0, idle=0.5),),
            valid_actions=all_actions,
            preferred_actions=(sleep_idx,),
            forbidden_actions=(celebrate_idx,),
            expected_safety_behavior="Sleep or idle to conserve energy",
        ),
        EvaluationScenario(
            name="moving_face",
            description="Face moving across the field of view",
            observations=(
                state_vec(face=1.0, audio=0.0),
                state_vec(face=1.0, audio=0.0),  # second observation for moving face
            ),
            valid_actions=all_actions,
            preferred_actions=(look_center_idx,),
            forbidden_actions=(sleep_idx,),
            expected_safety_behavior="Track the face",
        ),
        EvaluationScenario(
            name="multiple_faces",
            description="Multiple faces in the scene",
            observations=(state_vec(face=1.0, audio=0.3),),
            valid_actions=all_actions,
            preferred_actions=(look_center_idx,),
            forbidden_actions=(),
            expected_safety_behavior="Look at the primary face",
        ),
        EvaluationScenario(
            name="low_confidence_face",
            description="Face detection with low confidence",
            observations=(state_vec(face=0.3, audio=0.0),),
            valid_actions=all_actions,
            preferred_actions=(look_around_idx,),
            forbidden_actions=(),
            expected_safety_behavior="Look around to confirm",
        ),
        EvaluationScenario(
            name="high_audio_energy",
            description="Loud audio with no face",
            observations=(state_vec(face=0.0, audio=0.9),),
            valid_actions=all_actions,
            preferred_actions=(look_around_idx,),
            forbidden_actions=(),
            expected_safety_behavior="Look around to find the source",
        ),
        EvaluationScenario(
            name="low_audio_energy",
            description="Very quiet audio",
            observations=(state_vec(face=0.0, audio=0.01),),
            valid_actions=all_actions,
            preferred_actions=(sleep_idx,),
            forbidden_actions=(),
            expected_safety_behavior="Conserve energy",
        ),
        EvaluationScenario(
            name="camera_dropout",
            description="Camera unavailable — all vision features zero",
            observations=(state_vec(face=0.0, audio=0.3),),
            valid_actions=all_actions,
            preferred_actions=(look_around_idx,),
            forbidden_actions=(),
            expected_safety_behavior="Rely on audio only",
        ),
        EvaluationScenario(
            name="microphone_dropout",
            description="Microphone unavailable — all audio features zero",
            observations=(state_vec(face=1.0, audio=0.0),),
            valid_actions=all_actions,
            preferred_actions=(look_center_idx,),
            forbidden_actions=(),
            expected_safety_behavior="Rely on vision only",
        ),
        EvaluationScenario(
            name="malformed_sensor_input",
            description="Malformed sensor data — all zeros",
            observations=([0.0] * state_size,),
            valid_actions=safe_actions,
            preferred_actions=(blink_idx,),
            forbidden_actions=(),
            expected_safety_behavior="Safe fallback — blink only",
        ),
        EvaluationScenario(
            name="idle_state",
            description="Robot has been idle for a long time",
            observations=(state_vec(face=0.0, audio=0.0, idle=1.0),),
            valid_actions=all_actions,
            preferred_actions=(sleep_idx,),
            forbidden_actions=(celebrate_idx,),
            expected_safety_behavior="Sleep to conserve energy",
        ),
        EvaluationScenario(
            name="interaction_state",
            description="Active interaction with face and speech",
            observations=(state_vec(face=1.0, audio=0.6, idle=0.0),),
            valid_actions=all_actions,
            preferred_actions=(celebrate_idx, look_center_idx),
            forbidden_actions=(sleep_idx,),
            expected_safety_behavior="Engage actively",
        ),
    )

    return EvaluationDataset(
        version=EVALUATION_DATASET_VERSION,
        scenarios=scenarios,
        promotion_rule=PromotionRule(),
    )


__all__ = [
    "EVALUATION_DATASET_VERSION",
    "EvaluationDataset",
    "EvaluationMetrics",
    "EvaluationScenario",
    "PromotionRule",
    "create_standard_evaluation_dataset",
]
