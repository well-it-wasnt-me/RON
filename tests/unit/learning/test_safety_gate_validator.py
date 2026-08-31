"""Phase 8: the non-mutating ``SafetyGateValidator`` adapter + servo defense.

The :class:`SafetyGateValidator` adapts the :class:`SafetyGate` to the
:class:`ActionValidator` protocol used by :meth:`ActionLearner.select_action`
during practice. It must be **non-mutating** (static layer only) so the
per-candidate loop does not inflate the rate window or stall on cooldown.
The full, mutating :meth:`SafetyGate.validate` is re-applied later, once, on
the single chosen action.

These tests also cover the defence-in-depth servo-range check: an action
that resolves to an out-of-range servo angle is rejected before it reaches
hardware, regardless of what the caller passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot.learning.action_learning import ActionSpace, deskbot_action_space
from robot.learning.safety_gate import SafetyGate, SafetyGateValidator


@pytest.fixture
def action_space() -> ActionSpace:
    return deskbot_action_space()


@pytest.fixture
def gate(action_space: ActionSpace) -> SafetyGate:
    return SafetyGate(
        action_space=action_space,
        cooldown_s=0.5,
        servo_limits={
            "pan": (-90.0, 90.0),
            "tilt": (-30.0, 30.0),
            "left_arm": (0.0, 180.0),
            "right_arm": (0.0, 180.0),
        },
    )


def _state() -> np.ndarray:
    return np.zeros(91, dtype=float)


def test_validator_allows_valid_actions(
    gate: SafetyGate, action_space: ActionSpace
) -> None:
    """Wave (13) and look_left (0) are valid, non-servo actions."""
    validator = SafetyGateValidator(gate)
    assert validator.is_valid(13, _state(), action_space) is True
    assert validator.is_valid(0, _state(), action_space) is True


def test_validator_rejects_out_of_range_index(
    gate: SafetyGate, action_space: ActionSpace
) -> None:
    validator = SafetyGateValidator(gate)
    assert validator.is_valid(999, _state(), action_space) is False
    assert validator.is_valid(-1, _state(), action_space) is False


def test_validator_allows_in_range_arm_action(
    gate: SafetyGate, action_space: ActionSpace
) -> None:
    """move_left_arm (14) default angle 90 is within [0, 180]."""
    validator = SafetyGateValidator(gate)
    assert validator.is_valid(14, _state(), action_space) is True


def test_validator_is_non_mutating(
    gate: SafetyGate, action_space: ActionSpace
) -> None:
    """Calling is_valid many times does not consume cooldown / rate state.

    After a tight loop of non-mutating checks, the first real ``validate``
    must still be allowed (cooldown was not pre-armed), and only the second
    ``validate`` within the cooldown window is rejected.
    """
    validator = SafetyGateValidator(gate)
    for _ in range(20):
        assert validator.is_valid(13, _state(), action_space) is True

    first = gate.validate(13, state=None)
    assert first.allowed  # validator did not arm cooldown

    second = gate.validate(13, state=None)
    assert not second.allowed  # now the mutating gate enforces cooldown


def test_gate_rejects_out_of_range_arm_angle(gate: SafetyGate) -> None:
    """Defence-in-depth: an over-range arm angle is rejected statically."""
    assert gate.is_valid(14, params={"angle": 999.0}) is False


def test_gate_rejects_negative_arm_angle(gate: SafetyGate) -> None:
    assert gate.is_valid(14, params={"angle": -10.0}) is False


def test_gate_allows_override_in_range(gate: SafetyGate) -> None:
    assert gate.is_valid(14, params={"angle": 45.0}) is True


def test_validator_protocol_matches_safety_gate(
    gate: SafetyGate, action_space: ActionSpace
) -> None:
    """The validator and a direct is_valid agree on a valid action."""
    validator = SafetyGateValidator(gate)
    assert validator.is_valid(15, _state(), action_space) == gate.is_valid(15)
