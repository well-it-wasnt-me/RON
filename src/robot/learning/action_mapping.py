"""Map runtime :class:`BehaviorAction` instances to :class:`ActionSpace` identities.

The behaviour layer produces :class:`BehaviorAction` data objects (continuous look
coordinates, raw servo moves, blinks, …) while the learning system reasons over a
fixed, discrete :class:`ActionSpace`.  This module bridges the two: it identifies which
configured action - if any - a given runtime action corresponds to.

Only actions that resolve to a registered action-space identity can open a learning
transition (see :class:`robot.learning.transition.TransitionStore`).  Actions with no
action-space entry (e.g. a raw arm-servo move) return ``None`` and are **not** recorded
as transitions - they have no meaningful action identity for learning.

The gaze/servo normalisation deliberately matches
:meth:`robot.learning.state_encoder.StateEncoder._encode_servos`
(``(angle - 90) / 90``) so that a labelled action and the encoded next-state it produces
agree on direction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from robot.behavior.actions import (
    BehaviorAction,
    CelebrateAction,
    ChangeEmotionAction,
    LookAroundAction,
    MoveArmAction,
    RequestBlinkAction,
    RequestLookAction,
    RequestServoMoveAction,
    RequestSleepAction,
    SetStateAction,
    SpeakAction,
    WaveAction,
)
from robot.behavior.state_machine import RobotState
from robot.events.events import EmotionName
from robot.learning.action_learning import ActionSpace, LearningAction

# Magnitude (in normalised [-1, 1] gaze space) below which a look is treated as "centre".
_LOOK_THRESHOLD = 0.25

#: Servo names whose motion is a gaze direction (horizontal).  These map to the
#: ``look_left``/``look_right``/``look_center`` actions.
_PAN_SERVOS = frozenset({"pan", "head_pan"})

#: Servo names whose motion is a gaze direction (vertical).  These map to the
#: ``look_up``/``look_down``/``look_center`` actions.
_TILT_SERVOS = frozenset({"tilt", "head_tilt"})

#: Arm servo names that resolve to a learnable ``move_left_arm``/``move_right_arm``
#: action. Non-arm servos (pan/tilt) resolve via gaze quantisation; unknown servos
#: have no action-space identity.
_ARM_SERVOS = frozenset({"left_arm", "right_arm"})


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


def _name_for_blink(action: RequestBlinkAction) -> str | None:
    return "blink" if (action.left and action.right) else "wink"


def _name_for_servo_move(action: RequestServoMoveAction) -> str | None:
    # A raw servo move resolves to a gaze action for pan/tilt, or to a
    # learnable arm action for left_arm/right_arm. Other servos -> None.
    if action.servo in _ARM_SERVOS:
        return "move_left_arm" if action.servo == "left_arm" else "move_right_arm"
    return _servo_to_look(action.servo, action.angle)


def _name_for_move_arm(action: MoveArmAction) -> str | None:
    if action.servo == "left_arm":
        return "move_left_arm"
    if action.servo == "right_arm":
        return "move_right_arm"
    return None


def _name_for_look(action: RequestLookAction) -> str | None:
    return _quantize_look(action.x, action.y)


def _name_for_look_around(action: LookAroundAction) -> str | None:
    return "look_around"


def _name_for_celebrate(action: CelebrateAction) -> str | None:
    return "celebrate"


def _name_for_sleep(action: RequestSleepAction) -> str | None:
    return "sleep"


def _name_for_wave(action: WaveAction) -> str | None:
    return "wave"


def _name_for_speak(action: SpeakAction) -> str | None:
    return "speak"


def _name_for_change_emotion(action: ChangeEmotionAction) -> str | None:
    return "change_emotion"


def _name_for_set_state(action: SetStateAction) -> str | None:
    return "set_state"


#: ``isinstance`` dispatch from action type to a name resolver. Each resolver
#: returns the action-space name or ``None`` (unmappable). This keeps
#: :func:`behavior_action_to_name` branch-free. The resolver parameter is
#: typed ``Any`` because each resolver narrows to its concrete action type
#: internally (isinstance dispatch guarantees the right type at runtime).
_NAME_RESOLVERS: tuple[tuple[type[BehaviorAction], Callable[[Any], str | None]], ...] = (
    (RequestBlinkAction, _name_for_blink),
    (RequestLookAction, _name_for_look),
    (LookAroundAction, _name_for_look_around),
    (CelebrateAction, _name_for_celebrate),
    (RequestSleepAction, _name_for_sleep),
    (RequestServoMoveAction, _name_for_servo_move),
    (MoveArmAction, _name_for_move_arm),
    (WaveAction, _name_for_wave),
    (SpeakAction, _name_for_speak),
    (ChangeEmotionAction, _name_for_change_emotion),
    (SetStateAction, _name_for_set_state),
)


def behavior_action_to_name(action: BehaviorAction) -> str | None:
    """Resolve a runtime action to its action-space name, or ``None``.

    Returns ``None`` when the action does not correspond to any registered
    :class:`ActionSpace` action (e.g. an arm-servo move to an unknown servo).
    Such actions execute normally but do not produce a learning transition.
    """
    for action_type, resolver in _NAME_RESOLVERS:
        if isinstance(action, action_type):
            return resolver(action)
    return None


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


def _as_float(value: object, default: float) -> float:
    """Coerce a params value to float, falling back to ``default``."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def action_index_to_behavior_action(
    index: int,
    action_space: ActionSpace,
    override_params: dict[str, object] | None = None,
) -> BehaviorAction | None:
    """Resolve an action-space index back to a concrete :class:`BehaviorAction`.

    The reverse of :func:`behavior_action_to_index`. Used by both the
    demonstration path (execute a human-specified action) and the policy
    proposal path (execute the Q-selected action) so that every learned
    action flows through the same :class:`ActionExecutor`.

    Parameters
    ----------
    index:
        The action-space index to resolve.
    action_space:
        The action space to resolve against.
    override_params:
        Optional params merged over the action's registered defaults
        (e.g. a teaching demonstration overriding the ``text``/``emotion``/
        ``angle``). Validation still applies to the merged values.

    Returns
    -------
    BehaviorAction | None
        The concrete action, or ``None`` if the index is out of range, the
        action name is unknown, or a validated field (emotion, state, arm
        angle) is invalid - so callers can fall back safely.
    """
    if not (0 <= index < action_space.size):
        return None
    action: LearningAction = action_space.get(index)
    params: dict[str, object] = dict(action.params)
    if override_params:
        params.update(override_params)

    builder = _ACTION_BUILDERS.get(action.name)
    if builder is None:
        return None
    return builder(params)


def _build_wave(params: dict[str, object]) -> BehaviorAction:
    return WaveAction()


def _build_speak(params: dict[str, object]) -> BehaviorAction:
    return SpeakAction(text=str(params.get("text", "hello")))


def _build_change_emotion(params: dict[str, object]) -> BehaviorAction | None:
    emotion = str(params.get("emotion", "happy"))
    try:
        EmotionName(emotion)
    except ValueError:
        return None
    return ChangeEmotionAction(
        emotion=emotion, intensity=_as_float(params.get("intensity", 1.0), 1.0)
    )


def _build_set_state(params: dict[str, object]) -> BehaviorAction | None:
    state = str(params.get("state", "idle"))
    try:
        RobotState(state)
    except ValueError:
        return None
    return SetStateAction(state=state)


def _build_arm(servo: str) -> Callable[[dict[str, object]], BehaviorAction | None]:
    def _build(params: dict[str, object]) -> BehaviorAction | None:
        angle = _as_float(params.get("angle", 90.0), 90.0)
        if not (0.0 <= angle <= 180.0):
            return None
        return MoveArmAction(servo=servo, angle=angle)

    return _build


def _build_look(params: dict[str, object]) -> BehaviorAction:
    return RequestLookAction(
        x=_as_float(params.get("x", 0.0), 0.0),
        y=_as_float(params.get("y", 0.0), 0.0),
    )


def _build_blink(params: dict[str, object]) -> BehaviorAction:
    return RequestBlinkAction(left=True, right=True, speed=_as_float(params.get("speed", 1.0), 1.0))


def _build_wink(params: dict[str, object]) -> BehaviorAction:
    return RequestBlinkAction(
        left=True, right=False, speed=_as_float(params.get("speed", 1.5), 1.5)
    )


def _build_celebrate(params: dict[str, object]) -> BehaviorAction:
    return CelebrateAction(intensity=_as_float(params.get("intensity", 0.7), 0.7))


def _build_sleep(params: dict[str, object]) -> BehaviorAction:
    return RequestSleepAction(duration_s=_as_float(params.get("duration_s", 30.0), 30.0))


def _build_look_around(params: dict[str, object]) -> BehaviorAction:
    return LookAroundAction(points=int(_as_float(params.get("points", 3), 3.0)))


#: Dispatch table from action-space name to a builder that turns the merged
#: params (registered defaults + optional overrides) into a concrete
#: :class:`BehaviorAction`. Builders may return ``None`` when a validated
#: field (emotion, state, arm angle) is invalid.
_ACTION_BUILDERS: dict[str, Callable[[dict[str, object]], BehaviorAction | None]] = {
    "wave": _build_wave,
    "speak": _build_speak,
    "change_emotion": _build_change_emotion,
    "set_state": _build_set_state,
    "move_left_arm": _build_arm("left_arm"),
    "move_right_arm": _build_arm("right_arm"),
    "look_left": _build_look,
    "look_right": _build_look,
    "look_center": _build_look,
    "look_up": _build_look,
    "look_down": _build_look,
    "blink": _build_blink,
    "wink": _build_wink,
    "celebrate": _build_celebrate,
    "sleep": _build_sleep,
    "look_around": _build_look_around,
}


__all__ = [
    "action_index_to_behavior_action",
    "behavior_action_to_index",
    "behavior_action_to_name",
]
