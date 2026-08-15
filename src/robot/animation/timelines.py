"""Timeline primitives.

An :class:`Animation` is a coroutine that runs for a finite duration and
yields progress events. Compositions (:class:`Parallel`, :class:`Queue`) make
it easy to express complex motions declaratively.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from robot.animation.easing import Easing, linear


class AnimationPhase(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FINISHED = "finished"


Callback = Callable[[float], Any]


@dataclass(slots=True)
class Animation:
    """Base class for all timeline primitives."""

    duration_s: float = 0.0
    easing: Easing = field(default=linear)

    _phase: AnimationPhase = field(default=AnimationPhase.PENDING, init=False, repr=False)
    _cancelled: bool = field(default=False, init=False, repr=False)

    @property
    def phase(self) -> AnimationPhase:
        return self._phase

    def cancel(self) -> None:
        if self._phase is AnimationPhase.RUNNING:
            self._phase = AnimationPhase.INTERRUPTED
        self._cancelled = True

    async def run(self) -> None:
        """Subclasses implement this."""
        raise NotImplementedError


@dataclass(slots=True)
class Wait(Animation):
    """A no-op delay - useful for sequencing."""

    duration_s: float = 0.0

    async def run(self) -> None:
        import anyio

        self._phase = AnimationPhase.RUNNING
        try:
            if self.duration_s > 0:
                await anyio.sleep(self.duration_s)
        finally:
            self._phase = AnimationPhase.INTERRUPTED if self._cancelled else AnimationPhase.FINISHED


@dataclass(slots=True)
class Tween(Animation):
    """Interpolate a value over time and feed it to a callback.

    The callback receives the eased progress each frame. The callback may be
    sync or async.
    """

    from_value: float = 0.0
    to_value: float = 1.0
    on_update: Callback = field(default=lambda _v: None)

    async def run(self) -> None:
        import anyio

        from robot.animation.scheduler import _invoke_callback

        self._phase = AnimationPhase.RUNNING
        try:
            if self.duration_s <= 0:
                await _invoke_callback(self.on_update, self.to_value)
                return
            steps = max(1, int(self.duration_s * 60))
            step_s = self.duration_s / steps
            for i in range(steps + 1):
                if self._cancelled:
                    return
                t = i / steps
                value = self.from_value + (self.to_value - self.from_value) * self.easing(t)
                await _invoke_callback(self.on_update, value)
                await anyio.sleep(step_s)
        finally:
            self._phase = AnimationPhase.INTERRUPTED if self._cancelled else AnimationPhase.FINISHED


@dataclass(slots=True)
class Parallel(Animation):
    """Run several animations concurrently.

    ``duration_s`` is the wall-clock budget for the parallel group; the
    children determine their own per-step behaviour.
    """

    duration_s: float = 0.0
    children: list[Animation] = field(default_factory=list)

    async def run(self) -> None:
        import anyio

        self._phase = AnimationPhase.RUNNING
        try:
            async with anyio.create_task_group() as tg:
                for child in self.children:
                    tg.start_soon(_guarded, child)
        finally:
            self._phase = AnimationPhase.INTERRUPTED if self._cancelled else AnimationPhase.FINISHED


@dataclass(slots=True)
class Queue(Animation):
    """Run several animations sequentially."""

    duration_s: float = 0.0
    children: list[Animation] = field(default_factory=list)

    async def run(self) -> None:
        self._phase = AnimationPhase.RUNNING
        try:
            for child in self.children:
                if self._cancelled:
                    return
                await child.run()
        finally:
            self._phase = AnimationPhase.INTERRUPTED if self._cancelled else AnimationPhase.FINISHED


class Timeline:
    """Builder-style helper for declaring animations."""

    def __init__(self, animations: list[Animation] | None = None) -> None:
        self._items: list[Animation] = list(animations or [])

    def tween(
        self,
        from_value: float,
        to_value: float,
        duration_s: float,
        easing: Easing = linear,
        on_update: Callback | None = None,
    ) -> Timeline:
        self._items.append(
            Tween(
                duration_s=duration_s,
                easing=easing,
                from_value=from_value,
                to_value=to_value,
                on_update=on_update if on_update is not None else (lambda _v: None),
            )
        )
        return self

    def wait(self, duration_s: float) -> Timeline:
        self._items.append(Wait(duration_s=duration_s))
        return self

    def parallel(self, *children: Animation) -> Timeline:
        max_dur = max((c.duration_s for c in children), default=0.0)
        self._items.append(Parallel(duration_s=max_dur, children=list(children)))
        return self

    def queue(self, *children: Animation) -> Timeline:
        total = sum(c.duration_s for c in children)
        self._items.append(Queue(duration_s=total, children=list(children)))
        return self

    def build(self) -> Queue:
        total = sum(c.duration_s for c in self._items)
        return Queue(duration_s=total, children=list(self._items))

    async def run(self) -> None:
        await self.build().run()


async def _guarded(animation: Animation) -> None:
    with contextlib.suppress(Exception):
        await animation.run()


__all__ = [
    "Animation",
    "AnimationPhase",
    "Parallel",
    "Queue",
    "Timeline",
    "Tween",
    "Wait",
]
