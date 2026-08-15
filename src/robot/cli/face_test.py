"""``deskbot-face-test`` - prove the display + face rendering pipeline works.

Drives the FaceAnimator for a short burst and reports:
  * how many frames were produced
  * whether the face changed between emotions
  * whether frames reached the display backend (``frames_pushed``)

This is the cleanest way to validate the GC9A01A / circuitpython / gc9a01
display drivers without involving the conversation / wake-word stack.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace

from robot.config import load_settings
from robot.face.animator import FaceAnimator
from robot.face.emotions import EmotionEngine
from robot.face.renderer import FaceRenderer
from robot.hardware.displays.factory import DisplayFactory
from robot.logging import configure_logging, get_logger
from robot.utils.clock import SystemClock

_log = get_logger("cli.face_test")


async def _run(args: argparse.Namespace) -> int:
    settings = load_settings()
    configure_logging(settings)

    display = DisplayFactory(settings.displays).build()
    print(f"display: {type(display).__name__} ({display.width}x{display.height})")

    renderer = FaceRenderer(width=display.width, height=display.height)
    emotions = EmotionEngine(width=display.width, height=display.height)
    animator = FaceAnimator(
        renderer=renderer,
        display=display,
        clock=SystemClock(),
        emotions=emotions,
        theme=None,  # type: ignore[arg-type]
        fps=settings.displays.fps,
    )
    # Theme is required for rendering but isn't used by FaceAnimator.
    from robot.face.themes import VectorTheme

    animator = replace(animator, theme=VectorTheme())

    frames = []
    for emotion in ("neutral", "happy", "sad", "surprised", "happy"):
        animator.set_emotion(emotion)
        for _ in range(args.fps):  # ~1 second per emotion
            frame = animator.step(drift=False)
            frames.append(frame)
            await display.show(frame)

    print(f"produced: {len(frames)} frames")
    # frames_pushed is MockDisplay-specific
    frames_pushed = getattr(display, "frames_pushed", len(frames))
    print(f"display frames_pushed: {frames_pushed}")
    if frames_pushed < 1:
        print("FAIL: display never received a frame")
        return 1

    # Verify the face models differ between emotions.
    samples = {
        emo: frames[i * args.fps] for i, emo in enumerate(("neutral", "happy", "sad", "surprised"))
    }
    print(f"frame sizes (bytes): {[(k, len(v.pixels)) for k, v in samples.items()]}")

    neutral_pixels = samples["neutral"].pixels
    happy_pixels = samples["happy"].pixels
    sad_pixels = samples["sad"].pixels
    surprised_pixels = samples["surprised"].pixels

    def diff_fraction(a: bytes, b: bytes) -> float:
        if len(a) != len(b):
            return 1.0
        diff = sum(1 for x, y in zip(a, b, strict=False) if x != y)
        return diff / len(a)

    print(
        "frame difference (vs neutral): "
        f"happy={diff_fraction(neutral_pixels, happy_pixels):.3f} "
        f"sad={diff_fraction(neutral_pixels, sad_pixels):.3f} "
        f"surprised={diff_fraction(neutral_pixels, surprised_pixels):.3f}"
    )

    if diff_fraction(neutral_pixels, happy_pixels) < 0.01:
        print("FAIL: happy and neutral frames are identical")
        return 1
    if diff_fraction(neutral_pixels, sad_pixels) < 0.01:
        print("FAIL: sad and neutral frames are identical")
        return 1

    print("PASS: face animator produces differentiated frames")
    await display.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DeskBot face animator test")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings)
    rc = asyncio.run(_run(args))
    _log.info("face_test.done", returncode=rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
