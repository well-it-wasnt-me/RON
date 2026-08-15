"""Behavior primitives.

A :class:`Behavior` is a *recipe*: a list of steps to run, where each
step is either a face action, a body-language request, or a wait.
The runner walks the steps in order (some can be parallel) and returns
when the last step is done.

Behaviors are **declarative** - they describe *what* to do, not *how*.
The runner decides how to schedule the steps against the face engine
and the body-language engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robot.body_language.engine import BodyLanguageEngine
from robot.body_language.requests import BodyRequest
from robot.face.animator import FaceAnimator
from robot.logging import get_logger
from robot.utils.clock import Clock

_log = get_logger("behavior_library.behavior")


# ---------------------------------------------------------------------------
# Step types (tagged union)
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class FaceStep:
    """A face command: ``method(*args, duration_s=...)``."""

    method: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BodyStep:
    """A body-language request."""

    request: BodyRequest


@dataclass(slots=True, frozen=True)
class WaitStep:
    """A pause (seconds)."""

    seconds: float


@dataclass(slots=True, frozen=True)
class BehaviorStep:
    """One entry in a behavior's recipe.

    A step is one of:
    * :class:`FaceStep` - call a method on the :class:`FaceAnimator`.
    * :class:`BodyStep` - perform a :class:`BodyRequest` on the body.
    * :class:`WaitStep` - sleep.
    """

    name: str
    kind: str  # "face" | "body" | "wait"
    face: FaceStep | None = None
    body: BodyStep | None = None
    wait: WaitStep | None = None


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Behavior:
    """A named sequence of steps."""

    name: str
    steps: tuple[BehaviorStep, ...] = ()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class BehaviorRunner:
    """Run :class:`Behavior` objects against a face + body engine."""

    def __init__(
        self,
        face: FaceAnimator,
        body: BodyLanguageEngine,
        clock: Clock,
    ) -> None:
        self.face = face
        self.body = body
        self.clock = clock

    async def run(self, behavior: Behavior) -> None:
        _log.info("behavior.start", name=behavior.name, steps=len(behavior.steps))
        for step in behavior.steps:
            await self._run_step(step)
        _log.info("behavior.done", name=behavior.name)

    async def _run_step(self, step: BehaviorStep) -> None:
        if step.kind == "face" and step.face is not None:
            method = getattr(self.face, step.face.method, None)
            if method is None:
                _log.warning("behavior.unknown_face_method", method=step.face.method)
                return
            await _maybe_await(method(*step.face.args, **step.face.kwargs))
        elif step.kind == "body" and step.body is not None:
            await self.body.perform(step.body.request)
        elif step.kind == "wait" and step.wait is not None:
            await self.clock.sleep(step.wait.seconds)


async def _maybe_await(value: Any) -> None:
    """Await ``value`` if it is awaitable, otherwise do nothing."""
    if hasattr(value, "__await__"):
        await value


# ---------------------------------------------------------------------------
# Step builders (ergonomic API)
# ---------------------------------------------------------------------------
def face(method: str, *args: Any, **kwargs: Any) -> BehaviorStep:
    return BehaviorStep(
        name=method, kind="face", face=FaceStep(method=method, args=args, kwargs=kwargs)
    )


def body(request: BodyRequest) -> BehaviorStep:
    return BehaviorStep(name=request.name, kind="body", body=BodyStep(request=request))


def wait(seconds: float) -> BehaviorStep:
    return BehaviorStep(name=f"wait_{seconds:.2f}s", kind="wait", wait=WaitStep(seconds=seconds))


# ---------------------------------------------------------------------------
# Pre-built behaviors
# ---------------------------------------------------------------------------
from robot.body_language.requests import (  # noqa: E402
    ArmsOpen,
    ArmsRelax,
    HeadNod,
    LookLeft,
    LookRight,
    Wave,
)


def greeting() -> Behavior:
    """Smile, blink, raise arms, tilt head."""
    return Behavior(
        name="greeting",
        steps=(
            face("set_emotion", "happy"),
            face("smile_grow"),
            face("blink"),
            body(ArmsOpen(amount=15.0)),
            face("look_left"),
            wait(0.2),
            face("look_center"),
            face("look_right"),
            wait(0.2),
            face("look_center"),
            body(ArmsRelax()),
        ),
    )


def thinking() -> Behavior:
    """Look up, brows furrowed, subtle head movement."""
    return Behavior(
        name="thinking",
        steps=(
            face("set_emotion", "thinking"),
            face("look_up"),
            face("eyebrow_raise", 0.3),
            body(LookLeft(amount=10.0)),
            wait(0.5),
            body(LookRight(amount=10.0)),
            wait(0.5),
            body(LookLeft(amount=10.0)),
            wait(0.3),
            face("look_center"),
            face("reset"),
        ),
    )


def listening() -> Behavior:
    """Eyes focused, head tilted slightly, arms relaxed."""
    return Behavior(
        name="listening",
        steps=(
            face("set_emotion", "curious"),
            face("look_left", 0.1),
            body(HeadNod(amplitude=8.0, duration_s=0.6)),
            wait(0.4),
            face("look_center"),
            face("reset"),
        ),
    )


def sleeping() -> Behavior:
    """Eyelids close, breathing animation, head lowers, arms rest."""
    return Behavior(
        name="sleeping",
        steps=(
            face("set_emotion", "sleepy"),
            body(LookLeft(amount=0.0, duration_s=0.0)),
            wait(0.5),
            face("bounce"),
            wait(0.3),
            face("bounce"),
            wait(0.3),
            face("bounce"),
        ),
    )


def excited() -> Behavior:
    """Wider eyes, quick blink, bounce, small arm movement."""
    return Behavior(
        name="excited",
        steps=(
            face("set_emotion", "excited"),
            face("bounce"),
            face("blink"),
            body(Wave(amplitude=30.0, duration_s=0.8)),
            face("bounce"),
            face("blink"),
            wait(0.2),
            face("bounce"),
        ),
    )


def surprised() -> Behavior:
    """Wide eyes, open mouth, small jump."""
    return Behavior(
        name="surprised",
        steps=(
            face("set_emotion", "surprised"),
            face("bounce"),
            wait(0.5),
            face("reset"),
        ),
    )


__all__ = [
    "Behavior",
    "BehaviorRunner",
    "BehaviorStep",
    "BodyStep",
    "FaceStep",
    "WaitStep",
    "body",
    "excited",
    "face",
    "greeting",
    "listening",
    "sleeping",
    "surprised",
    "thinking",
    "wait",
]
