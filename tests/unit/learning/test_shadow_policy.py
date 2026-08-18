"""Tests for the shadow-mode policy controller.

Phase 6: Shadow-Mode Policy.

Tests prove:
- in shadow mode, the model predicts but the deterministic controller executes
- every decision is logged with all required fields
- policy agreement, disagreement, and unsafe proposals are tracked
- the learned policy has zero authority to execute hardware actions
- the mode can be switched (off/shadow/assist/active)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from robot.learning.action_learning import ActionSpace, deskbot_action_space
from robot.learning.shadow_policy import (
    PolicyMode,
    ShadowPolicyController,
)


@pytest.fixture
def action_space() -> ActionSpace:
    return deskbot_action_space()


@pytest.fixture
def deterministic_policy() -> Callable[[np.ndarray], int]:
    """A simple deterministic controller: always looks center."""

    def _det(state: np.ndarray) -> int:
        return 2  # look_center

    return _det


@pytest.fixture
def model_policy() -> Callable[[np.ndarray], tuple[int, list[float]]]:
    """A learned policy that sometimes agrees, sometimes doesn't."""

    def _model(state: np.ndarray) -> tuple[int, list[float]]:
        # Return action 0 (look_left) with some scores
        return 0, [0.9, 0.1, 0.5, 0.3, 0.2, 0.1, 0.1, 0.4, 0.05, 0.15]

    return _model


@pytest.fixture
def safety_checker() -> Callable[[int, np.ndarray, ActionSpace], tuple[bool, str]]:
    """Reject the 'sleep' action (index 8)."""

    def _check(action_idx: int, state: np.ndarray, space: object) -> tuple[bool, str]:
        if action_idx == 8:  # sleep
            return False, "sleep not allowed in shadow mode"
        return True, "ok"

    return _check


@pytest.fixture
def controller(
    action_space: ActionSpace,
    deterministic_policy: Callable[[np.ndarray], int],
    model_policy: Callable[[np.ndarray], tuple[int, list[float]]],
    safety_checker: Callable[[int, np.ndarray, ActionSpace], tuple[bool, str]],
) -> ShadowPolicyController:
    return ShadowPolicyController(
        action_space=action_space,
        deterministic_policy=deterministic_policy,
        model_policy=model_policy,
        safety_checker=safety_checker,
        model_version="v1.0.0",
    )


# ========================================================================
# Shadow mode: model predicts, deterministic executes
# ========================================================================


class TestShadowMode:
    """In shadow mode, the model predicts but the deterministic controller executes."""

    def test_deterministic_action_returned(self, controller: ShadowPolicyController) -> None:
        """The returned action is the deterministic one, not the model's."""
        state = np.zeros(91, dtype=np.float64)
        action = controller.decide(state)
        # Deterministic policy always returns 2 (look_center)
        assert action == 2

    def test_model_action_logged_not_executed(self, controller: ShadowPolicyController) -> None:
        """The model's action is logged but not returned."""
        state = np.zeros(91, dtype=np.float64)
        controller.decide(state)
        entry = controller.log_entries[-1]
        # Model predicted 0 (look_left), but deterministic returned 2
        assert entry.model_action == 0
        assert entry.deterministic_action == 2
        assert not entry.agreement

    def test_log_has_all_required_fields(self, controller: ShadowPolicyController) -> None:
        """Every log entry has all required fields."""
        state = np.zeros(91, dtype=np.float64)
        controller.decide(state, observation_id="obs-001")
        entry = controller.log_entries[-1]
        assert entry.timestamp_ns > 0
        assert entry.observation_id == "obs-001"
        assert entry.deterministic_action >= 0
        assert entry.model_action >= 0
        assert isinstance(entry.model_scores, list)
        assert entry.model_version == "v1.0.0"
        assert entry.safety_result == "ok"
        assert entry.inference_latency_ms >= 0.0
        assert isinstance(entry.agreement, bool)

    def test_agreement_when_actions_match(
        self,
        action_space: ActionSpace,
        deterministic_policy: Callable[[np.ndarray], int],
        safety_checker: Callable[[int, np.ndarray, ActionSpace], tuple[bool, str]],
    ) -> None:
        """Agreement is logged when model and deterministic agree."""

        # Model also returns 2 (look_center)
        def model(state: np.ndarray) -> tuple[int, list[float]]:
            return (2, [0.1, 0.1, 0.9])

        controller = ShadowPolicyController(
            action_space=action_space,
            deterministic_policy=deterministic_policy,
            model_policy=model,
            safety_checker=safety_checker,
        )
        state = np.zeros(91, dtype=np.float64)
        controller.decide(state)
        entry = controller.log_entries[-1]
        assert entry.agreement is True
        assert controller.metrics.agreement_count == 1
        assert controller.metrics.disagreement_count == 0

    def test_disagreement_tracked(self, controller: ShadowPolicyController) -> None:
        """Disagreement is tracked in metrics."""
        state = np.zeros(91, dtype=np.float64)
        controller.decide(state)
        assert controller.metrics.disagreement_count == 1
        assert controller.metrics.agreement_count == 0


# ========================================================================
# Safety checking
# ========================================================================


class TestSafetyChecking:
    """Model actions are checked for safety in shadow mode."""

    def test_unsafe_proposal_logged(
        self,
        action_space: ActionSpace,
        deterministic_policy: Callable[[np.ndarray], int],
        safety_checker: Callable[[int, np.ndarray, ActionSpace], tuple[bool, str]],
    ) -> None:
        """An unsafe model proposal is logged as rejected."""

        def model(state: np.ndarray) -> tuple[int, list[float]]:
            return (8, [0.0] * 10)  # sleep — unsafe

        controller = ShadowPolicyController(
            action_space=action_space,
            deterministic_policy=deterministic_policy,
            model_policy=model,
            safety_checker=safety_checker,
        )
        state = np.zeros(91, dtype=np.float64)
        action = controller.decide(state)
        # Still returns the deterministic action (2)
        assert action == 2
        entry = controller.log_entries[-1]
        assert "rejected" in entry.safety_result
        assert controller.metrics.unsafe_proposals == 1

    def test_model_error_handled(
        self,
        action_space: ActionSpace,
        deterministic_policy: Callable[[np.ndarray], int],
        safety_checker: Callable[[int, np.ndarray, ActionSpace], tuple[bool, str]],
    ) -> None:
        """Model errors are handled gracefully."""

        def broken_model(state: np.ndarray) -> tuple[int, list[float]]:
            raise RuntimeError("model crashed")

        controller = ShadowPolicyController(
            action_space=action_space,
            deterministic_policy=deterministic_policy,
            model_policy=broken_model,
            safety_checker=safety_checker,
        )
        state = np.zeros(91, dtype=np.float64)
        action = controller.decide(state)
        assert action == 2  # deterministic still works
        entry = controller.log_entries[-1]
        assert "error" in entry.safety_result


# ========================================================================
# Metrics
# ========================================================================


class TestMetrics:
    """Shadow metrics track agreement, latency, and safety."""

    def test_metrics_after_multiple_decisions(self, controller: ShadowPolicyController) -> None:
        """Metrics accumulate correctly over multiple decisions."""
        state = np.zeros(91, dtype=np.float64)
        for _ in range(10):
            controller.decide(state)
        m = controller.metrics
        assert m.total_decisions == 10
        assert m.agreement_count + m.disagreement_count == 10
        assert m.avg_latency_ms >= 0.0

    def test_metrics_to_dict(self, controller: ShadowPolicyController) -> None:
        """Metrics can be serialised to a dict."""
        state = np.zeros(91, dtype=np.float64)
        controller.decide(state)
        d = controller.metrics.to_dict()
        assert "total_decisions" in d
        assert "agreement_rate" in d
        assert "model_version" in d

    def test_reset_metrics(self, controller: ShadowPolicyController) -> None:
        """Reset clears metrics and logs."""
        state = np.zeros(91, dtype=np.float64)
        controller.decide(state)
        assert len(controller.log_entries) > 0
        controller.reset_metrics()
        assert len(controller.log_entries) == 0
        assert controller.metrics.total_decisions == 0


# ========================================================================
# Mode switching
# ========================================================================


class TestModeSwitching:
    """The mode can be switched between off/shadow/assist/active."""

    def test_off_mode_no_model_inference(
        self,
        action_space: ActionSpace,
        deterministic_policy: Callable[[np.ndarray], int],
        model_policy: Callable[[np.ndarray], tuple[int, list[float]]],
        safety_checker: Callable[[int, np.ndarray, ActionSpace], tuple[bool, str]],
    ) -> None:
        """In off mode, the model is not consulted."""
        controller = ShadowPolicyController(
            action_space=action_space,
            deterministic_policy=deterministic_policy,
            model_policy=model_policy,
            safety_checker=safety_checker,
            mode=PolicyMode.OFF,
        )
        state = np.zeros(91, dtype=np.float64)
        action = controller.decide(state)
        assert action == 2
        # No log entries in off mode (model not consulted)
        assert len(controller.log_entries) == 0

    def test_set_mode(self, controller: ShadowPolicyController) -> None:
        """Mode can be changed at runtime."""
        controller.set_mode(PolicyMode.OFF)
        assert controller.mode.value == "off"
        controller.set_mode(PolicyMode.SHADOW)
        assert controller.mode.value == "shadow"


# ========================================================================
# Definition of done: zero authority
# ========================================================================


class TestZeroAuthority:
    """The learned policy has zero authority to execute hardware actions."""

    def test_model_never_controls_hardware(self, controller: ShadowPolicyController) -> None:
        """The returned action is always the deterministic one."""
        state = np.zeros(91, dtype=np.float64)
        for _ in range(100):
            action = controller.decide(state)
            assert action == 2  # always deterministic

    def test_no_model_still_works(
        self, action_space: ActionSpace, deterministic_policy: Callable[[np.ndarray], int]
    ) -> None:
        """Without a model, the controller still works."""
        controller = ShadowPolicyController(
            action_space=action_space,
            deterministic_policy=deterministic_policy,
            model_policy=None,
            mode=PolicyMode.SHADOW,
        )
        state = np.zeros(91, dtype=np.float64)
        action = controller.decide(state)
        assert action == 2  # deterministic works without a model
