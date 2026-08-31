"""Map runtime :class:`BehaviorAction` instances to :class:`ActionSpace` identities.

The behaviour layer produces :class:`BehaviorAction` data objects (continuous look
coordinates, raw servo moves, blinks, …) while the learning system reasons over a
fixed, discrete :class:`ActionSpace`.  This module bridges the two: it identifies which
configured action — if any — a given runtime action corresponds to.

Only actions that resolve to a registered action-space identity can open a learning
transition (see :class:`robot.learning.transition.TransitionStore`).  Actions with no
action-space entry (e.g. a raw arm-servo move) return ``None`` and are **not** recorded
as transitions — they have no meaningful action identity for learning.

The gaze/servo normalisation deliberately matches
:meth:`robot.learning.state_encoder.StateEncoder._encode_servos`
(``(angle - 90) / 90``) so that a labelled action and the encoded next-state it produces
agree on direction.
"""

from __future__ import annotations

from robot.behavior.actions import (
    BehaviorAction,
    CelebrateAction,
    LookAroundAction,
    RequestBlinkAction,
    RequestLookAction,
    RequestServoMoveAction,
    RequestSleepAction,
)
from robot.learning.action_learning import ActionSpace

# Magnitude (in normalised [-1, 1] gaze space) below which a look is treated as "centre".
_LOOK_THRESHOLD = 0.25

#: Servo names whose motion is a gaze direction (horizontal).  These map to the
#: ``look_left``/``look_right``/``look_center`` actions.
_PAN_SERVOS = frozenset({"pan", "head_pan"})

#: Servo names whose motion is a gaze direction (vertical).  These map to the
#: ``look_up``/``look_down``/``look_center`` actions.
_TILT_SERVOS = frozenset({"tilt", "head_tilt"})


def _quantize_look(x: float, y: float) -> str:
    """Pick the nearest discrete look action for a continuous gaze coordinate."""
    if abs(x) < _LOOK_THRESHOLD and abs(y) < _LOOK_THRESHOLD:
        return "look_center"
    if abs(x) >= abs(y):
        return "look_left" if x < 0 else "look_right"
    return "look_up" if y < 0 else "look_down"


def _servo_to_look(servo: str, angle: float) -> str | None:
    """Map a servo move to a gaze action, or ``None`` for non-gaze servos.

    Uses the same normalisation as the state encoder (``(angle - 90) / 90``) so the
    labelled direction agrees with the encoded servo position.  Pan/tilt moves are
    projected onto the horizontal/vertical gaze axis and quantised via
    :func:`_quantize_look`, so a move near centre resolves to ``look_center``.
    """
    norm = (float(angle) - 90.0) / 90.0
    if servo in _PAN_SERVOS:
        return _quantize_look(norm, 0.0)
    if servo in _TILT_SERVOS:
        return _quantize_look(0.0, norm)
    # Non-gaze servos (arms, …) have no action-space identity.
    return None


def behavior_action_to_name(action: BehaviorAction) -> str | None:
    """Resolve a runtime action to its action-space name, or ``None``.

    Returns ``None`` when the action does not correspond to any registered
    :class:`ActionSpace` action (e.g. an arm-servo move).  Such actions execute
    normally but do not produce a learning transition.
    """
    name: str | None
    if isinstance(action, RequestBlinkAction):
        name = "blink" if (action.left and action.right) else "wink"
    elif isinstance(action, RequestLookAction):
        name = _quantize_look(action.x, action.y)
    elif isinstance(action, LookAroundAction):
        name = "look_around"
    elif isinstance(action, CelebrateAction):
        name = "celebrate"
    elif isinstance(action, RequestSleepAction):
        name = "sleep"
    elif isinstance(action, RequestServoMoveAction):
        name = _servo_to_look(action.servo, action.angle)
    else:
        name = None
    return name


def behavior_action_to_index(action: BehaviorAction, action_space: ActionSpace) -> int | None:
    """Resolve a runtime action to its action-space index, or ``None``.

    ``None`` is returned both for unmappable actions and for actions whose resolved
    name is not present in ``action_space`` (defensive against mismatched spaces).
    """
    name = behavior_action_to_name(action)
    if name is None:
        return None
    try:
        return action_space.get_by_name(name).index
    except KeyError:
        return None


__all__ = ["behavior_action_to_index", "behavior_action_to_name"]
