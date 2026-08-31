"""Behavior actions - pure data describing what the robot wants to do.

The behavior engine produces :class:`BehaviorAction` instances; a separate
executor (or, in tests, an in-memory recorder) turns them into hardware
commands. The action layer never imports hardware directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ActionName = Literal[
    "blink",
    "look",
    "servo_move",
    "sleep",
    "celebrate",
    "look_around",
    "wave",
    "speak",
    "change_emotion",
    "set_state",
    "move_arm",
]


@dataclass(slots=True, frozen=True)
class BehaviorAction:
    """Base class for behavior actions."""

    name: ActionName
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RequestBlinkAction(BehaviorAction):
    """A request to blink. Pass ``left=False`` to wink."""

    name: ActionName = "blink"
    left: bool = True
    right: bool = True
    speed: float = 1.0


@dataclass(slots=True, frozen=True)
class RequestLookAction(BehaviorAction):
    """A request to glance at a position."""

    name: ActionName = "look"
    x: float = 0.0
    y: float = 0.0
    duration_s: float = 0.3


@dataclass(slots=True, frozen=True)
class RequestServoMoveAction(BehaviorAction):
    """A request to move a named servo to a specific angle."""

    name: ActionName = "servo_move"
    servo: str = ""
    angle: float = 0.0
    duration_s: float = 0.4


@dataclass(slots=True, frozen=True)
class RequestSleepAction(BehaviorAction):
    """A request to enter the SLEEPING state for a given duration."""

    name: ActionName = "sleep"
    duration_s: float = 30.0


@dataclass(slots=True, frozen=True)
class CelebrateAction(BehaviorAction):
    """A request to express a short celebratory reaction."""

    name: ActionName = "celebrate"
    intensity: float = 1.0


@dataclass(slots=True, frozen=True)
class LookAroundAction(BehaviorAction):
    """A request to perform a quick look-around sequence."""

    name: ActionName = "look_around"
    points: int = 3


@dataclass(slots=True, frozen=True)
class WaveAction(BehaviorAction):
    """A request to wave the right arm (a short up/center/up/center sequence).

    This is a *learnable* behaviour action registered in the action space.
    The execution layer drives the ``right_arm`` servo through the wave
    sequence; the action itself carries no parameters.
    """

    name: ActionName = "wave"


@dataclass(slots=True, frozen=True)
class SpeakAction(BehaviorAction):
    """A request to speak the given text via TTS.

    The learner reasons over *when* to speak (the discrete action index);
    the text content is execution metadata, not a per-utterance learnable
    dimension.
    """

    name: ActionName = "speak"
    text: str = "hello"


@dataclass(slots=True, frozen=True)
class ChangeEmotionAction(BehaviorAction):
    """A request to change the robot's facial emotion.

    ``emotion`` must be a valid :class:`EmotionName` value; the execution
    layer validates it before publishing ``EmotionChanged``.
    """

    name: ActionName = "change_emotion"
    emotion: str = "happy"
    intensity: float = 1.0


@dataclass(slots=True, frozen=True)
class SetStateAction(BehaviorAction):
    """A request to set the robot behaviour state directly.

    ``state`` must be a valid :class:`RobotState` value. Like the existing
    tool-executor ``set_state`` path, this publishes ``StateChanged``
    directly rather than going through the state machine's legality
    transition (a warning is logged on illegal targets at execution time).
    """

    name: ActionName = "set_state"
    state: str = "idle"


@dataclass(slots=True, frozen=True)
class MoveArmAction(BehaviorAction):
    """A request to move a named arm servo (``left_arm``/``right_arm``).

    The resolved action-space name (``move_left_arm``/``move_right_arm``)
    depends on ``servo``. The angle is validated against the servo's
    hardware range at execution time (``ServoError`` is raised out of
    range, as for any servo move).
    """

    name: ActionName = "move_arm"
    servo: str = "right_arm"
    angle: float = 90.0
    duration_s: float = 0.4


__all__ = [
    "ActionName",
    "BehaviorAction",
    "CelebrateAction",
    "ChangeEmotionAction",
    "LookAroundAction",
    "MoveArmAction",
    "RequestBlinkAction",
    "RequestLookAction",
    "RequestServoMoveAction",
    "RequestSleepAction",
    "SetStateAction",
    "SpeakAction",
    "WaveAction",
]
