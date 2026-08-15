"""``deskbot-doctor`` diagnostic command."""

from __future__ import annotations

import argparse
import contextlib
import math
import platform
import struct
import sys
from pathlib import Path

from robot.config import AppSettings, load_settings
from robot.hardware.sensors.usb_microphone import UsbMicrophone
from robot.lifecycle.degradation import DegradationRegistry
from robot.logging import get_logger

_log = get_logger("cli.doctor")


def main() -> int:
    """Run DeskBot diagnostics and return a process exit code."""
    args = _parse_args()
    settings = load_settings()

    if args.microphone:
        return _run_microphone_diagnostic(settings, duration_s=args.duration_s)
    if args.audio:
        return _run_audio_diagnostic(
            settings,
            duration_s=args.duration_s,
            frequency_hz=args.frequency_hz,
        )

    _print_summary(settings)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="deskbot-doctor", description="DeskBot diagnostics")
    parser.add_argument(
        "--microphone",
        action="store_true",
        help="enumerate input devices and capture a short microphone sample",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="play a short speaker test through the configured audio backend",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=1.5,
        help="sample/playback duration in seconds for diagnostic actions",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=440.0,
        help="speaker test tone frequency for --audio",
    )
    return parser.parse_args()


def _print_summary(settings: AppSettings) -> None:
    print("DeskBot doctor")
    print("--------------")
    print(f"Python:       {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Platform:     {platform.platform()}")
    print(f"Machine:      {platform.machine()}")
    print(f"Env:          {settings.env}")
    print(f"Log level:    {settings.log_level}")
    print(f"Timezone:     {settings.timezone}")
    print(f"Hardware:     {settings.hardware}")
    print()
    print("Backends")
    print("--------")
    print(
        f"Display:      {settings.displays.backend}  "
        f"({settings.displays.width}x{settings.displays.height} "
        f"@ {settings.displays.fps}fps, rotation={settings.displays.rotation})"
    )
    print(f"  bus:         {settings.displays.bus}")
    print(f"  device:      {settings.displays.device}")
    print(f"Servos:       {settings.servos.backend}")
    print(
        f"  pan:         BCM {settings.servos.head_pan.gpio_pin}  "
        f"(neutral={settings.servos.head_pan.center_angle_deg}, "
        f"min={settings.servos.head_pan.min_angle_deg}, "
        f"max={settings.servos.head_pan.max_angle_deg})"
    )
    print(
        f"  tilt:        BCM {settings.servos.head_tilt.gpio_pin}  "
        f"(neutral={settings.servos.head_tilt.center_angle_deg}, "
        f"min={settings.servos.head_tilt.min_angle_deg}, "
        f"max={settings.servos.head_tilt.max_angle_deg})"
    )
    print(
        f"  left_arm:    BCM {settings.servos.left_arm.gpio_pin}  "
        f"(neutral={settings.servos.left_arm.center_angle_deg}, "
        f"min={settings.servos.left_arm.min_angle_deg}, "
        f"max={settings.servos.left_arm.max_angle_deg})"
    )
    print(
        f"  right_arm:   BCM {settings.servos.right_arm.gpio_pin}  "
        f"(neutral={settings.servos.right_arm.center_angle_deg}, "
        f"min={settings.servos.right_arm.min_angle_deg}, "
        f"max={settings.servos.right_arm.max_angle_deg})"
    )
    print(f"SPI bus:      /dev/spidev{settings.displays.bus}.{settings.displays.device}")
    print(f"SPI check:    {spi_status()}")
    print()
    print("AI providers")
    print("------------")
    print(f"LLM provider: {settings.llm.provider} ({settings.llm.model})")
    print(f"TTS provider: {settings.tts.provider}")
    print(f"STT provider: {settings.stt.provider}")
    print(f"Wake-word:    {settings.wakeword.provider} ({settings.wakeword.phrase!r})")
    print(
        f"Perception:   {'enabled' if settings.perception.enabled else 'disabled'} "
        f"(idle {settings.perception.idle_scan_interval_s}s, "
        f"curious {settings.perception.curious_scan_interval_s}s, "
        f"max {settings.perception.max_faces} faces, "
        f"threshold {settings.perception.score_threshold})"
    )
    print(
        f"Microphone:   input={settings.microphone.input_device!r}, "
        f"{settings.microphone.sample_rate} Hz, {settings.microphone.channels}ch"
    )
    print(
        f"Audio:        backend={settings.audio.backend}, "
        f"output={settings.audio.output_device!r}, "
        f"{settings.audio.sample_rate} Hz, {settings.audio.channels}ch"
    )
    print(f"Assets dir:   {Path('assets').resolve()}")
    print()
    print("Diagnostic commands")
    print("-------------------")
    print("deskbot-doctor --microphone")
    print("deskbot-doctor --audio")


def _run_microphone_diagnostic(settings: AppSettings, *, duration_s: float) -> int:
    print("DeskBot microphone diagnostic")
    print("-----------------------------")
    try:
        inventory = UsbMicrophone.list_input_devices()
    except Exception as exc:
        print(f"Failed to enumerate microphones: {exc}")
        return 1

    print(f"Configured input_device: {settings.microphone.input_device!r}")
    print(f"PortAudio default input: {inventory['default_input_device']}")
    print("Available input devices:")
    for device in inventory["devices"]:
        marker = " (default)" if device["is_default"] else ""
        print(
            f"  [{device['index']}] {device['name']} | "
            f"channels={device['max_input_channels']} | "
            f"default_sr={device['default_sample_rate']}{marker}"
        )

    try:
        result = UsbMicrophone.diagnose_capture(
            input_device=settings.microphone.input_device,
            sample_rate=settings.microphone.sample_rate,
            channels=settings.microphone.channels,
            frame_ms=settings.microphone.frame_ms,
            duration_s=duration_s,
        )
    except Exception as exc:
        print()
        print(f"Capture failed: {exc}")
        return 1

    print()
    print(f"Resolved device index:   {result['resolved_device_index']}")
    print(f"Resolved device name:    {result['resolved_device_name']}")
    print(f"Max input channels:      {result['max_input_channels']}")
    print(f"Default sample rate:     {result['default_sample_rate']}")
    print(f"Requested sample rate:   {result['requested_sample_rate']}")
    print(f"Actual stream rate:      {result['actual_stream_sample_rate']}")
    print(f"Channels:                {result['channels']}")
    print(f"Frame samples:           {result['frame_samples']}")
    print(f"Observed chunks:         {result['observed_chunks']}")
    print(f"Observed bytes:          {result['observed_bytes']}")
    print(f"Non-zero audio:          {result['nonzero_audio']}")
    print(
        f"RMS min/max/avg:         {result['rms_min']} / {result['rms_max']} / {result['rms_avg']}"
    )
    print(f"Min/max sample:          {result['min_sample']} / {result['max_sample']}")
    print(f"PortAudio overflows:     {result['overflows']}")
    return 0 if result["nonzero_audio"] else 1


def _run_audio_diagnostic(
    settings: AppSettings,
    *,
    duration_s: float,
    frequency_hz: float,
) -> int:
    print("DeskBot audio diagnostic")
    print("------------------------")
    print(f"Configured audio backend: {settings.audio.backend}")
    print(f"Configured output device: {settings.audio.output_device!r}")

    if settings.audio.backend == "mock":
        print("Audio backend is mock; no physical playback will occur.")
        return 1

    from robot.app import _build_audio
    from robot.interfaces.audio import AudioBuffer

    degradation = DegradationRegistry()
    audio = _build_audio(settings, degradation)
    report = {entry.component: entry for entry in degradation.report()}
    audio_entry = report.get("audio")
    print(f"Active audio backend:     {type(audio).__name__}")
    print(f"Audio degraded:           {audio_entry.status == 'degraded' if audio_entry else False}")
    if audio_entry and audio_entry.error:
        print(f"Audio error:              {audio_entry.error}")
        return 1

    pcm = _generate_tone_pcm(
        frequency_hz=frequency_hz,
        duration_s=duration_s,
        sample_rate=settings.audio.sample_rate,
        volume=0.4,
    )
    buffer = AudioBuffer(
        pcm=pcm,
        sample_rate=settings.audio.sample_rate,
        channels=settings.audio.channels,
    )
    import anyio

    try:
        anyio.run(audio.play, buffer)
    except Exception as exc:
        print(f"Playback failed: {exc}")
        return 1
    finally:
        with contextlib.suppress(Exception):
            anyio.run(audio.close)
    print(
        f"Played {duration_s:.2f}s test tone at {frequency_hz:.1f} Hz "
        f"({buffer.sample_rate} Hz, {buffer.channels} channel(s))."
    )
    return 0


def _generate_tone_pcm(
    *,
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    volume: float,
) -> bytes:
    n_frames = max(1, int(sample_rate * duration_s))
    samples = []
    for index in range(n_frames):
        phase = 2.0 * math.pi * frequency_hz * (index / sample_rate)
        samples.append(int(max(-1.0, min(1.0, math.sin(phase) * volume)) * 32767))
    return struct.pack(f"<{len(samples)}h", *samples)


def _is_mock(obj: object, mock_class_names: frozenset[str]) -> bool:
    """Check if *obj* is a mock by class name."""
    from robot.hardware.displays.mock_display import MockDisplay
    from robot.hardware.sensors.mock_camera import MockCamera
    from robot.hardware.sensors.mock_microphone import MockMicrophone

    if isinstance(obj, (MockDisplay, MockCamera, MockMicrophone)):
        return True
    return type(obj).__name__ in mock_class_names


_MOCK_CLASS_NAMES = frozenset({"MockDisplay", "MockCamera", "MockMicrophone"})


def _hardware_banner(
    settings: AppSettings,
    display: object,
    *,
    microphone: object | None = None,
    camera: object | None = None,
) -> None:
    """Log a one-line summary of which backends are real vs mock."""
    disp_real = not _is_mock(display, _MOCK_CLASS_NAMES)
    mic_real = microphone is not None and not _is_mock(microphone, _MOCK_CLASS_NAMES)
    cam_real = camera is not None and not _is_mock(camera, _MOCK_CLASS_NAMES)

    _log.info(
        "hardware.active",
        backend=settings.hardware,
        display_real=disp_real,
        display_backend=type(display).__name__,
        microphone_real=mic_real,
        microphone_class=type(microphone).__name__ if microphone else None,
        camera_real=cam_real,
        camera_class=type(camera).__name__ if camera else None,
    )
    if not disp_real and settings.hardware == "real":
        _log.error("hardware.fallback", component="display", backend=type(display).__name__)
    if not mic_real and microphone is not None and settings.hardware == "real":
        _log.error("hardware.fallback", component="microphone", backend=type(microphone).__name__)
    if not cam_real and camera is not None and settings.hardware == "real":
        _log.error("hardware.fallback", component="camera", backend=type(camera).__name__)


def spi_status() -> str:
    """Return a short status string describing the SPI bus availability."""
    candidates = [Path("/dev/spidev0.0"), Path("/dev/spidev0.1")]
    found = [str(path) for path in candidates if path.exists()]
    if not found:
        return "NOT FOUND (enable SPI in `sudo raspi-config` and reboot)"
    return "available: " + ", ".join(found)


if __name__ == "__main__":
    raise SystemExit(main())
