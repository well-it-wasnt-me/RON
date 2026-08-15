"""Tests for the animation framework."""

from __future__ import annotations

import asyncio

import anyio
import pytest

from robot.animation.easing import (
    builtin_easings,
    ease_in_quad,
    ease_out_bounce,
    linear,
)
from robot.animation.scheduler import AnimationScheduler
from robot.animation.timelines import (
    AnimationPhase,
    Parallel,
    Queue,
    Timeline,
    Tween,
    Wait,
)


def test_easing_endpoints() -> None:
    for name, fn in builtin_easings.items():
        assert fn(0.0) == pytest.approx(0.0, abs=1e-6) or fn(0.0) == pytest.approx(1.0, abs=1e-6)
        # ease_out_bounce and ease_out_elastic don't start at 0; accept either endpoint.
        assert -0.5 <= fn(0.5) <= 1.5, name  # elastic can overshoot
        assert fn(1.0) == pytest.approx(1.0, abs=1e-3), name


def test_linear_easing() -> None:
    assert linear(0.0) == 0.0
    assert linear(0.5) == 0.5
    assert linear(1.0) == 1.0


def test_ease_in_quad() -> None:
    assert ease_in_quad(0.5) == pytest.approx(0.25, abs=1e-6)


def test_ease_out_bounce_within_range() -> None:
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = ease_out_bounce(t)
        assert 0.0 <= v <= 1.0


async def test_tween_runs_and_finishes() -> None:
    values: list[float] = []

    def on_update(v: float) -> None:
        values.append(v)

    tween = Tween(duration_s=0.05, from_value=0.0, to_value=1.0, on_update=on_update)
    await tween.run()
    assert tween.phase is AnimationPhase.FINISHED
    assert values and values[0] == pytest.approx(0.0, abs=0.05)
    assert values[-1] == pytest.approx(1.0, abs=0.05)


async def test_wait_finishes() -> None:
    w = Wait(duration_s=0.01)
    await w.run()
    assert w.phase is AnimationPhase.FINISHED


async def test_queue_runs_children_in_order() -> None:
    log: list[str] = []
    q = Queue(
        duration_s=0.02,
        children=[
            Tween(duration_s=0.01, from_value=0, to_value=1, on_update=lambda v: log.append("a")),
            Tween(duration_s=0.01, from_value=0, to_value=1, on_update=lambda v: log.append("b")),
        ],
    )
    await q.run()
    # Queue runs a to completion, then b
    a_idxs = [i for i, v in enumerate(log) if v == "a"]
    b_idxs = [i for i, v in enumerate(log) if v == "b"]
    assert a_idxs and b_idxs
    assert max(a_idxs) < min(b_idxs)


async def test_parallel_runs_children_concurrently() -> None:
    log: list[str] = []
    p = Parallel(
        duration_s=0.02,
        children=[
            Wait(duration_s=0.01),
            Wait(duration_s=0.01),
        ],
    )

    async def tag(s: str) -> None:
        await anyio.sleep(0.005)
        log.append(s)

    p2 = Parallel(
        duration_s=0.02,
        children=[
            Tween(duration_s=0.01, from_value=0, to_value=1, on_update=lambda _v: tag("a")),
            Tween(duration_s=0.01, from_value=0, to_value=1, on_update=lambda _v: tag("b")),
        ],
    )
    await p.run()
    await p2.run()
    assert set(log) == {"a", "b"}


async def test_cancel_marks_interrupted() -> None:
    tween = Tween(duration_s=10.0, from_value=0.0, to_value=1.0, on_update=lambda v: None)
    task = asyncio.create_task(tween.run())
    await asyncio.sleep(0.01)
    tween.cancel()
    await task
    assert tween.phase is AnimationPhase.INTERRUPTED


async def test_scheduler_runs_periodic_jobs() -> None:
    scheduler = AnimationScheduler()
    ticks: list[int] = []
    counter = {"n": 0}

    def tick(_v: float) -> None:
        counter["n"] += 1
        ticks.append(counter["n"])

    async with scheduler.running():
        scheduler.schedule(tick, interval_s=0.001, name="t")
        await asyncio.sleep(0.05)
    assert ticks and ticks[-1] >= 2


async def test_timeline_builder() -> None:
    log: list[float] = []
    t = (
        Timeline()
        .tween(from_value=0.0, to_value=1.0, duration_s=0.01, on_update=log.append)
        .wait(0.0)
    )
    await t.run()
    assert log == pytest.approx([0.0, 1.0], abs=0.05)
