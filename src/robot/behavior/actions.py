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


__all__ = [
    "ActionName",
    "BehaviorAction",
    "CelebrateAction",
    "LookAroundAction",
    "RequestBlinkAction",
    "RequestLookAction",
    "RequestServoMoveAction",
    "RequestSleepAction",
]
