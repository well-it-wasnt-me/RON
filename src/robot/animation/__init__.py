"""Timeline-based animation framework."""

from robot.animation.easing import Easing, builtin_easings
from robot.animation.scheduler import AnimationScheduler, ScheduledTask
from robot.animation.timelines import (
    Animation,
    AnimationPhase,
    Parallel,
    Queue,
    Timeline,
    Tween,
    Wait,
)

__all__ = [
    "Animation",
    "AnimationPhase",
    "AnimationScheduler",
    "Easing",
    "Parallel",
    "Queue",
    "ScheduledTask",
    "Timeline",
    "Tween",
    "Wait",
    "builtin_easings",
]
