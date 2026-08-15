"""``deskbot-simulate`` - run the full robot stack against the mock display.

This is the recommended way to verify the new face engine, body-language
engine, and behavior library on a developer laptop with **zero
hardware**. The output is a single in-memory frame per frame; the
last frame is written to disk as a PNG so the user can inspect it.
"""

from __future__ import annotations

import argparse
import asyncio
import struct
import zlib
from pathlib import Path

from robot.behavior_library.behavior import (
    BehaviorRunner,
    excited as behavior_excited,
    greeting as behavior_greeting,
    listening as behavior_listening,
    sleeping as behavior_sleeping,
    thinking as behavior_thinking,
)
from robot.body_language.requests import (
    ArmsOpen,
    ArmsRelax,
    BodyRequest,
    Celebrate,
    Greet,
    HeadNod,
    LookLeft,
    LookRight,
    Shrug,
    Wave,
)
from robot.face.themes import get_theme
from robot.logging import configure_logging, get_logger
from robot.simulation.driver import SimulationDriver
from robot.utils.clock import SystemClock

_log = get_logger("cli.simulate")


# ---------------------------------------------------------------------------
# Minimal PNG writer (no external deps)
# ---------------------------------------------------------------------------
def _write_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    """Write a 24-bit RGB PNG file (no external deps)."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    # Add a filter byte (0 = None) to every row
    raw = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))
    compressed = zlib.compress(raw, 9)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


# ---------------------------------------------------------------------------
# Demo script
# ---------------------------------------------------------------------------
async def _run_demo(driver: SimulationDriver, *, fps: int, theme_name: str, output: Path) -> None:
    driver.theme = get_theme(theme_name)
    driver.face.theme = driver.theme
    runner = BehaviorRunner(face=driver.face, body=driver.body, clock=SystemClock())

    # 1. Start at neutral with relaxed body
    driver.face.set_emotion("neutral")
    driver.body.perform_sync(ArmsRelax())
    await asyncio.sleep(1.0 / fps)

    # 2. Cycle through the behavior library
    sequence = [
        behavior_greeting(),
        behavior_thinking(),
        behavior_listening(),
        behavior_sleeping(),
        behavior_excited(),
    ]
    for behavior in sequence:
        _log.info("simulate.behavior", name=behavior.name)
        await runner.run(behavior)
        await asyncio.sleep(0.3)

    # 3. Ad-hoc body requests so the stick figure moves
    requests: list[BodyRequest] = [
        Wave(),
        HeadNod(),
        LookLeft(),
        LookRight(),
        Celebrate(),
        Shrug(),
        ArmsOpen(),
        Greet(),
        ArmsRelax(),
    ]
    for request in requests:
        _log.info("simulate.body", name=request.name)
        await driver.body.perform(request)
        await asyncio.sleep(0.3)

    # Render one final frame and save it
    frame = driver.step()
    _write_png(output, frame.width, frame.height, frame.pixels)
    _log.info("simulate.saved", path=str(output), width=frame.width, height=frame.height)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeskBot simulator (no hardware)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--size", type=int, default=240, help="face size (square)")
    parser.add_argument(
        "--theme",
        default="minimal",
        choices=["minimal", "cute", "pixel", "retro_lcd", "wireframe"],
    )
    parser.add_argument("--output", type=Path, default=Path("simulate.png"), help="output PNG path")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from robot.config import AppSettings

    configure_logging(AppSettings(log_level=args.log_level))
    driver = SimulationDriver(
        width=args.size, height=int(args.size * 1.33), face_size=args.size, fps=args.fps
    )
    asyncio.run(_run_demo(driver, fps=args.fps, theme_name=args.theme, output=args.output))


if __name__ == "__main__":
    main()
