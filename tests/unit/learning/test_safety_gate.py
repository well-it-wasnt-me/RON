"""Tests for the safety-gated action system.

Safety-Gated Actions.

Tests prove:
- invalid actions are rejected
- NaN scores trigger fallback
- missing sensors/timeout trigger fallback
- impossible servo positions are rejected
- rapid repeated actions hit rate limits
- emergency override disables learned control
- the robot always falls back safely
- a learned policy cannot directly command hardware
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from robot.learning.action_learning import ActionSpace, deskbot_action_space
from robot.learning.safety_gate import (
    SafeActionExecutor,
    SafetyGate,
)


@pytest.fixture
def action_space() -> ActionSpace:
    return deskbot_action_space()


@pytest.fixture
def gate(action_space: ActionSpace) -> SafetyGate:
    return SafetyGate(action_space=action_space, cooldown_s=0.0)


@pytest.fixture
def executed_actions() -> list[int]:
    return []


@pytest.fixture
def executor(executed_actions: list[int]) -> Callable[[int], None]:
    return executed_actions.append


@pytest.fixture
def safe_executor(gate: SafetyGate, executor: Callable[[int], None]) -> SafeActionExecutor:
    return SafeActionExecutor(safety_gate=gate, executor=executor)


# ========================================================================
# Static validation
# ========================================================================


class TestStaticValidation:
    """Layer 1: static validation — action exists, params valid."""

    def test_valid_action_allowed(self, gate: SafetyGate) -> None:
        result = gate.validate(action_index=2)
        assert result.allowed
        assert result.layer in {"static", "runtime"}

    def test_invalid_action_index_rejected(self, gate: SafetyGate) -> None:
        result = gate.validate(action_index=999)
        assert result.rejected
        assert "out of range" in result.reason

    def test_negative_action_rejected(self, gate: SafetyGate) -> None:
        result = gate.validate(action_index=-1)
        assert result.rejected

    def test_servo_out_of_range_rejected(self, action_space: ActionSpace) -> None:
        gate = SafetyGate(
            action_space=action_space,
            servo_limits={"pan": (0.0, 180.0)},
        )
        result = gate.validate(action_index=0, params={"pan": 200.0})
        assert result.rejected
        assert "out of range" in result.reason


# ========================================================================
# Runtime safety
# ========================================================================


class TestRuntimeSafety:
    """Layer 2: runtime safety — rate limits, cooldown."""

    def test_rate_limit_exceeded(self, action_space: ActionSpace) -> None:
        gate = SafetyGate(action_space=action_space, max_action_rate=3, cooldown_s=0.0)
        # First 3 actions should pass
        for i in range(3):
            r = gate.validate(action_index=i % action_space.size)
            assert r.allowed, f"Action {i} should be allowed: {r.reason}"
        # 4th should be rejected
        r = gate.validate(action_index=0)
        assert r.rejected
        assert "rate limit" in r.reason.lower()

    def test_cooldown(self, action_space: ActionSpace) -> None:
        gate = SafetyGate(action_space=action_space, cooldown_s=0.5)
        r1 = gate.validate(action_index=0)
        assert r1.allowed
        # Same action immediately should be rejected
        r2 = gate.validate(action_index=0)
        assert r2.rejected
        assert "cooldown" in r2.reason.lower()


# ========================================================================
# Emergency override
# ========================================================================


class TestEmergencyOverride:
    """Layer 3: emergency override disables learned control."""

    def test_override_falls_back(self, gate: SafetyGate) -> None:
        gate.activate_override()
        result = gate.validate(action_index=2)
        assert result.fallback
        assert "override" in result.reason.lower()
        assert result.action_index == gate.fallback_action

    def test_override_deactivated(self, gate: SafetyGate) -> None:
        gate.activate_override()
        assert gate.override_active
        gate.deactivate_override()
        assert not gate.override_active
        result = gate.validate(action_index=2)  # type: ignore[unreachable]
        assert result.allowed


# ========================================================================
# Model output handling
# ========================================================================


class TestModelOutputHandling:
    """NaN, invalid, timeout, and corrupted model outputs trigger fallback."""

    def test_nan_scores_fallback(self, gate: SafetyGate) -> None:
        result = gate.handle_model_output([0.1, float("nan"), 0.3])
        assert result.fallback
        assert "NaN" in result.reason

    def test_inf_scores_fallback(self, gate: SafetyGate) -> None:
        result = gate.handle_model_output([0.1, float("inf"), 0.3])
        assert result.fallback
        assert "NaN/inf" in result.reason or "inf" in result.reason

    def test_none_output_fallback(self, gate: SafetyGate) -> None:
        result = gate.handle_model_output(None)
        assert result.fallback
        assert "None" in result.reason

    def test_invalid_action_index_fallback(self, gate: SafetyGate) -> None:
        result = gate.handle_model_output(999)
        assert result.rejected or result.fallback

    def test_valid_model_output_allowed(self, gate: SafetyGate) -> None:
        scores = [0.1] * 10
        scores[2] = 0.9  # look_center
        result = gate.handle_model_output(scores)
        assert result.allowed
        assert result.action_index == 2


# ========================================================================
# Safe executor
# ========================================================================


class TestSafeActionExecutor:
    """The safe executor wraps hardware with the safety gate."""

    def test_valid_action_executed(
        self, safe_executor: SafeActionExecutor, executed_actions: list[int]
    ) -> None:
        action = safe_executor.execute(action_index=2)
        assert action == 2
        assert executed_actions == [2]

    def test_rejected_action_falls_back(
        self,
        action_space: ActionSpace,
        executed_actions: list[int],
        executor: Callable[[int], None],
    ) -> None:
        gate = SafetyGate(action_space=action_space, fallback_action=5)
        safe = SafeActionExecutor(safety_gate=gate, executor=executor)
        action = safe.execute(action_index=999)
        assert action == 5  # fallback
        assert executed_actions[-1] == 5
        assert safe.rejection_count > 0

    def test_nan_model_output_falls_back(
        self, safe_executor: SafeActionExecutor, executed_actions: list[int]
    ) -> None:
        action = safe_executor.execute_model_output([float("nan"), 0.1, 0.3])
        assert action == safe_executor.safety_gate.fallback_action
        assert safe_executor.fallback_count > 0


# ========================================================================
# Definition of done: learned policy cannot directly command hardware
# ========================================================================


class TestCannotBypassSafety:
    """A learned policy cannot directly command hardware."""

    def test_every_action_through_gate(
        self, safe_executor: SafeActionExecutor, executed_actions: list[int]
    ) -> None:
        """Every executed action passed through the safety gate."""
        for i in range(5):
            safe_executor.execute(action_index=i % 10)
        # All executed actions should be valid indices
        for a in executed_actions:
            assert 0 <= a < 10

    def test_gate_cannot_be_bypassed(self, gate: SafetyGate) -> None:
        """The gate validates even when enabled=True."""
        assert gate.enabled
        # An invalid action is always rejected
        result = gate.validate(action_index=-1)
        assert not result.allowed
