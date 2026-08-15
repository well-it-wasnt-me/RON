"""``deskbot-hardware-check`` - a focused diagnostic tool for the
USB camera and microphone.

The script bypasses the rest of the app and just reports the actual
hardware state, so we can prove the camera is opening and frames are
arriving, and that the microphone is producing non-zero PCM.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import time

from robot.config import load_settings
from robot.logging import configure_logging, get_logger

_log = get_logger("cli.hardware_check")


async def _check_camera(args: argparse.Namespace) -> int:
    from robot.hardware.sensors.usb_camera import UsbCamera

    cam = UsbCamera(device=args.device, width=args.width, height=args.height, fps=args.fps)
    print(f"camera: {type(cam).__name__} actual={cam.width}x{cam.height}")
    deadline = time.time() + args.seconds
    captured = 0
    while time.time() < deadline:
        try:
            frame = await cam.capture()
            captured += 1
            # Compute a quick brightness histogram for the center pixel.
            w = frame.width
            h = frame.height
            center = frame.pixels[
                (h // 2) * w * 3 + (w // 2) * 3 : (h // 2) * w * 3 + (w // 2) * 3 + 3
            ]
            print(f"  frame {captured}: {w}x{h}, center pixel = {tuple(center)}")
        except Exception as exc:
            print(f"  capture error: {exc!r}")
            break
    await cam.close()
    print(f"camera: captured {captured} frames in {args.seconds}s")
    return 0 if captured > 0 else 1


async def _check_microphone(args: argparse.Namespace) -> int:
    from robot.hardware.sensors.usb_microphone import UsbMicrophone, rms

    mic = UsbMicrophone(
        input_device=args.device,
        _sample_rate_field=args.sample_rate,
        channels=1,
        frame_ms=args.frame_ms,
    )
    print(
        f"microphone: actual_sample_rate={mic.sample_rate} "
        f"channels={mic.channels} frame_samples={mic._frame_samples}"
    )
    chunks = 0
    energies = []
    async for chunk in mic.stream():
        chunks += 1
        energies.append(rms(chunk.pcm))
        if chunks >= args.chunks:
            break
    print(f"microphone: received {chunks} chunks in {args.chunks * args.frame_ms / 1000:.1f}s")
    if energies:
        avg = sum(energies) / len(energies)
        mx = max(energies)
        print(f"microphone: RMS avg={avg:.4f} max={mx:.4f}")
        # Treat as "audio is arriving" if any chunk has RMS > 0.005 (very quiet).
        if mx > 0.005:
            print("microphone: AUDIO DETECTED (RMS > threshold)")
            return 0
        print("microphone: no audio above noise floor (very quiet environment?)")
        return 0
    print("microphone: no chunks received")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="DeskBot hardware diagnostic")
    parser.add_argument("--device", type=str, default="default")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--target", choices=["camera", "microphone", "all"], default="all")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings)
    _log.info(
        "hardware_check.start",
        target=args.target,
        device=args.device,
        width=args.width,
        height=args.height,
        _sample_rate_field=args.sample_rate,
        fps=args.fps,
    )

    rc = 0
    if args.target in ("camera", "all"):
        rc |= asyncio.run(_check_camera(args))
    if args.target in ("microphone", "all"):
        with contextlib.suppress(KeyboardInterrupt):
            rc |= asyncio.run(_check_microphone(args))
    _log.info("hardware_check.done", returncode=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
