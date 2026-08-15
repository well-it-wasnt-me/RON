"""Eye engine demo CLI.

Run the full eye engine through every supported animation in a loop.
Designed to work with **no hardware** (uses the in-memory
:class:`MockDisplay`), so it is the recommended way to verify the eye
engine after a change.

Usage
-----

::

    deskbot-eye-demo                # 30 FPS, 240x240 mock display
    deskbot-eye-demo --fps 60 --size 128
    deskbot-eye-demo --no-drift     # disable idle gaze drift
"""

from __future__ import annotations

import argparse
import asyncio

import anyio

from robot.events.events import EmotionName
from robot.eye_engine.animator import EyeDisplayAnimator
from robot.eye_engine.renderer import EyeRenderer
from robot.hardware.displays.mock_display import MockDisplay
from robot.logging import configure_logging, get_logger
from robot.utils.clock import SystemClock

_log = get_logger("cli.eye_demo")


async def run_demo(
    *,
    fps: int,
    size: int,
    drift: bool,
    loops: int,
) -> None:
    display = MockDisplay(width=size, height=size)
    renderer = EyeRenderer(width=size, height=size)
    clock = SystemClock()

    animator = EyeDisplayAnimator(
        renderer=renderer,
        display=display,
        clock=clock,
        fps=fps,
        width=size,
        height=size,
    )

    _log.info(
        "eye_demo.start",
        fps=fps,
        size=size,
        drift=drift,
        frames_pushed=0,
    )

    frame_interval = 1.0 / fps
    sequence = _build_sequence()

    for loop_index in range(loops):
        for step in sequence:
            _apply_step(animator, step)
            frame = animator.step(drift=drift)
            await display.show(frame)
            _log.debug(
                "eye_demo.frame",
                loop=loop_index,
                step=step["name"],
                pushed=display.frames_pushed,
            )
            await anyio.sleep(frame_interval)

    total = display.frames_pushed
    _log.info("eye_demo.done", total_frames=total)
    await display.close()


def _build_sequence() -> list[dict[str, object]]:
    """Build the demo step list.

    Each step is a small dict describing what the animator should do.
    The duration is implicit: each step is held for ``fps // 6`` frames
    so the human eye can register the change (~150ms at 30fps).
    """
    return [
        {"name": "neutral", "kind": "emotion", "emotion": EmotionName.NEUTRAL},
        {"name": "look_left", "kind": "look", "x": -1.0, "y": 0.0, "duration_s": 0.3},
        {"name": "look_right", "kind": "look", "x": 1.0, "y": 0.0, "duration_s": 0.3},
        {"name": "look_up", "kind": "look", "x": 0.0, "y": -1.0, "duration_s": 0.3},
        {"name": "look_down", "kind": "look", "x": 0.0, "y": 1.0, "duration_s": 0.3},
        {"name": "look_center", "kind": "look", "x": 0.0, "y": 0.0, "duration_s": 0.3},
        {"name": "blink", "kind": "blink"},
        {"name": "double_blink", "kind": "double_blink"},
        {"name": "happy", "kind": "emotion", "emotion": EmotionName.HAPPY},
        {"name": "sleepy", "kind": "emotion", "emotion": EmotionName.SLEEPY},
        {"name": "surprised", "kind": "emotion", "emotion": EmotionName.SURPRISED},
        {"name": "angry", "kind": "emotion", "emotion": EmotionName.ANGRY},
        {"name": "sad", "kind": "emotion", "emotion": EmotionName.SAD},
        {"name": "curious", "kind": "emotion", "emotion": EmotionName.CURIOUS},
        {"name": "thinking", "kind": "emotion", "emotion": EmotionName.THINKING},
        {"name": "wink", "kind": "wink"},
        {"name": "back_to_neutral", "kind": "emotion", "emotion": EmotionName.NEUTRAL},
    ]


def _apply_step(animator: EyeDisplayAnimator, step: dict[str, object]) -> None:
    kind = step["kind"]
    if kind == "emotion":
        animator.set_emotion(step["emotion"])  # type: ignore[arg-type]
    elif kind == "look":
        animator.look(
            float(step["x"]),  # type: ignore[arg-type]
            float(step["y"]),  # type: ignore[arg-type]
            duration_s=float(step["duration_s"]),  # type: ignore[arg-type]
        )
    elif kind == "blink":
        animator.blink()
    elif kind == "double_blink":
        animator.double_blink()
    elif kind == "wink":
        animator.eye.wink()
    else:  # pragma: no cover
        raise ValueError(f"unknown demo step: {step!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeskBot eye engine demo")
    parser.add_argument("--fps", type=int, default=30, help="target frames per second")
    parser.add_argument("--size", type=int, default=240, help="display size (square)")
    parser.add_argument("--no-drift", action="store_true", help="disable idle drift")
    parser.add_argument("--loops", type=int, default=1, help="how many full cycles to run")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging()
    from robot.config import AppSettings

    configure_logging(AppSettings(log_level=args.log_level))
    asyncio.run(
        run_demo(
            fps=args.fps,
            size=args.size,
            drift=not args.no_drift,
            loops=max(1, args.loops),
        )
    )


if __name__ == "__main__":
    main()
