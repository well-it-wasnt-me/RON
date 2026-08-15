"""CLI command: ``deskbot-profile`` - run the robot and collect performance data.

Usage::

    deskbot-profile --duration 10 --output profile.json

This starts the DeskBot robot for the specified duration, collects profiling
data (frame budget, servo latency, event bus throughput), and writes a JSON
report to the given output path (or stdout if ``--output`` is ``-``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict

from robot.config import AppSettings, load_settings
from robot.logging import configure_logging, get_logger

_log = get_logger("cli.profile")


def main() -> None:
    """Entry point for the ``deskbot-profile`` CLI command."""
    parser = argparse.ArgumentParser(
        prog="deskbot-profile",
        description="Run DeskBot and collect performance profiling data.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration in seconds to run the robot (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output file path. Use '-' for stdout (default: -).",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to a YAML configuration file.",
    )
    args = parser.parse_args()

    # Build settings (optionally from a config file)
    import os

    if args.config_file:
        os.environ["DESKBOT_CONFIG_FILE"] = args.config_file

    settings = load_settings()
    configure_logging(settings)

    # Ensure profiling is enabled
    if not settings.performance.enabled:
        _log.info("profile.enabling_performance")
        settings.performance.enabled = True
        settings.performance.frame_profiling = True
        settings.performance.servo_profiling = True
        settings.performance.bus_profiling = True

    report = asyncio.run(_run_profile(settings, duration=args.duration))

    output_json = json.dumps(report, indent=2, default=str)
    if args.output == "-":
        print(output_json)
    else:
        from pathlib import Path

        with Path(args.output).open("w") as f:
            f.write(output_json)
        _log.info("profile.written", path=args.output)


async def _run_profile(settings: AppSettings, duration: float) -> dict[str, object]:
    """Run the robot for *duration* seconds and collect profiling data."""
    from robot.app import DeskBotApp

    app = DeskBotApp.from_settings(settings)

    # Retrieve profilers from app state
    frame_profiler = getattr(app, "_frame_profiler", None)
    servo_profiler = getattr(app, "_servo_profiler", None)
    bus_profiler = getattr(app, "_bus_profiler", None)

    _log.info("profile.starting", duration=duration)

    async with app.run():
        await asyncio.sleep(duration)

    _log.info("profile.completed")

    # Collect stats
    report: dict[str, object] = {
        "duration_seconds": duration,
    }

    if frame_profiler is not None:
        report["frames"] = asdict(frame_profiler.stats())
    else:
        report["frames"] = {"enabled": False}

    if servo_profiler is not None:
        report["servos"] = servo_profiler.stats()
    else:
        report["servos"] = {"enabled": False}

    if bus_profiler is not None:
        report["bus"] = bus_profiler.stats()
    else:
        report["bus"] = {"enabled": False}

    return report


if __name__ == "__main__":
    main()
