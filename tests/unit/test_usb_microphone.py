"""Tests for the USB microphone driver."""

from __future__ import annotations

import asyncio
import sys
import time
from array import array
from types import SimpleNamespace

import pytest

from robot.interfaces.microphone import AudioChunk


class _FakeInputStream:
    def __init__(
        self,
        *,
        samplerate: int,
        channels: int,
        blocksize: int,
        device: int,
        frames: list[array],  # type: ignore[type-arg]
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device
        self._frames = frames
        self._index = 0
        self._closed = False
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def read(self, _frame_samples: int) -> tuple[array, bool]:  # type: ignore[type-arg]
        if self._closed:
            raise RuntimeError("stream closed")
        frame = self._frames[self._index % len(self._frames)]
        self._index += 1
        time.sleep(0.002)
        return frame, False

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self._closed = True


class _FakeSoundDevice:
    def __init__(
        self,
        *,
        devices: list[dict[str, object]],
        default_input: int = 0,
        actual_sample_rate: int = 16_000,
        actual_channels: int = 1,
        frames: list[array] | None = None,  # type: ignore[type-arg]
        check_error: Exception | None = None,
    ) -> None:
        self._devices = devices
        self.default = SimpleNamespace(device=(default_input, 9))
        self._actual_sample_rate = actual_sample_rate
        self._actual_channels = actual_channels
        self._frames = frames or [array("h", [0] * 480)]
        self._check_error = check_error
        self.streams: list[_FakeInputStream] = []

    def query_devices(
        self, index: int | None = None, kind: str | None = None
    ) -> list[dict[str, object]] | dict[str, object]:
        if index is None:
            return list(self._devices)
        info = dict(self._devices[index])
        info["index"] = index
        del kind
        return info

    def check_input_settings(
        self,
        *,
        device: int,
        samplerate: int,
        channels: int,
        dtype: str,
    ) -> None:
        del device, samplerate, channels, dtype
        if self._check_error is not None:
            raise self._check_error

    def InputStream(  # noqa: N802 - matches sounddevice API
        self,
        *,
        samplerate: int,
        channels: int,
        dtype: str,
        blocksize: int,
        device: int,
    ) -> _FakeInputStream:
        del dtype
        stream = _FakeInputStream(
            samplerate=self._actual_sample_rate,
            channels=self._actual_channels,
            blocksize=blocksize,
            device=device,
            frames=self._frames,
        )
        self.streams.append(stream)
        return stream


def _install_sounddevice(monkeypatch: pytest.MonkeyPatch, fake_sd: _FakeSoundDevice) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)


def _devices() -> list[dict[str, object]]:
    return [
        {
            "name": "HDMI output",
            "max_input_channels": 0,
            "default_samplerate": 48_000,
        },
        {
            "name": "Built-in Microphone",
            "max_input_channels": 2,
            "default_samplerate": 48_000,
        },
        {
            "name": "USB Headset Mic",
            "max_input_channels": 1,
            "default_samplerate": 16_000,
        },
    ]


def test_usb_microphone_missing_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The friendly-error path is present in the source."""
    import inspect

    del monkeypatch
    from robot.hardware.sensors import usb_microphone

    source = inspect.getsource(usb_microphone.UsbMicrophone.__post_init__)
    assert "sounddevice is required" in source


def test_usb_microphone_rms_handles_short_buffer() -> None:
    from robot.hardware.sensors.usb_microphone import rms

    assert rms(b"") == 0.0
    assert rms(b"\x00") == 0.0


def test_usb_microphone_rms_zero_for_silence() -> None:
    from robot.hardware.sensors.usb_microphone import rms

    silence = b"\x00\x00" * 1024
    assert rms(silence) == 0.0


def test_usb_microphone_rms_positive_for_signal() -> None:
    import math
    import struct

    from robot.hardware.sensors.usb_microphone import rms

    pcm = struct.pack("<1024h", *([16384] * 1024))
    assert math.isclose(rms(pcm), 0.5, abs_tol=1e-4)


def test_default_resolves_portaudio_default_input(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot.hardware.sensors.usb_microphone import UsbMicrophone

    fake_sd = _FakeSoundDevice(devices=_devices(), default_input=2)
    _install_sounddevice(monkeypatch, fake_sd)

    mic = UsbMicrophone(input_device="default")
    assert mic.describe_selection()["resolved_device_index"] == 2


def test_numeric_device_index_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot.hardware.sensors.usb_microphone import UsbMicrophone

    fake_sd = _FakeSoundDevice(devices=_devices())
    _install_sounddevice(monkeypatch, fake_sd)

    mic = UsbMicrophone(input_device=1)
    assert mic.describe_selection()["resolved_device_name"] == "Built-in Microphone"


def test_invalid_device_gives_useful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot.hardware.sensors.usb_microphone import UsbMicrophone

    fake_sd = _FakeSoundDevice(devices=_devices())
    _install_sounddevice(monkeypatch, fake_sd)

    mic = UsbMicrophone(input_device="nope")
    with pytest.raises(RuntimeError, match="no microphone matching"):
        mic.describe_selection()


def test_device_with_no_input_channels_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot.hardware.sensors.usb_microphone import UsbMicrophone

    fake_sd = _FakeSoundDevice(devices=_devices())
    _install_sounddevice(monkeypatch, fake_sd)

    mic = UsbMicrophone(input_device=0)
    with pytest.raises(RuntimeError, match="has no input channels"):
        mic.describe_selection()


@pytest.mark.asyncio
async def test_audio_chunk_metadata_and_queue_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot.hardware.sensors.usb_microphone import UsbMicrophone

    frames = [
        array("h", [1000] * 480),
        array("h", [2000] * 480),
    ]
    fake_sd = _FakeSoundDevice(
        devices=_devices(),
        default_input=1,
        actual_sample_rate=48_000,
        actual_channels=1,
        frames=frames,
    )
    _install_sounddevice(monkeypatch, fake_sd)

    mic = UsbMicrophone(input_device="default", _sample_rate_field=16_000, channels=1, frame_ms=30)
    stream = mic.stream()
    chunk = await anext(stream)
    stats = mic.runtime_stats()

    assert isinstance(chunk, AudioChunk)
    assert chunk.sample_rate == 48_000
    assert chunk.channels == 1
    assert chunk.pcm != b""
    assert stats["chunks_received"] >= 1
    assert stats["chunks_enqueued"] >= 1
    assert stats["chunks_consumed"] >= 1
    assert stats["nonzero_chunks"] >= 1
    assert stats["chunks_dropped"] == 0
    assert stats["thread_alive"] is True

    await mic.close()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)

    assert mic.thread_alive is False
    assert fake_sd.streams[0].stopped is True


@pytest.mark.asyncio
async def test_queue_does_not_saturate_under_normal_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot.hardware.sensors.usb_microphone import UsbMicrophone

    fake_sd = _FakeSoundDevice(
        devices=_devices(),
        frames=[array("h", [300] * 480)],
    )
    _install_sounddevice(monkeypatch, fake_sd)

    mic = UsbMicrophone(input_device=1, _sample_rate_field=16_000, channels=1, frame_ms=30)
    stream = mic.stream()
    for _ in range(5):
        await anext(stream)
    stats = mic.runtime_stats()

    assert stats["chunks_dropped"] == 0
    assert stats["queue_size"] <= stats["queue_maxsize"]

    await mic.close()


def test_diagnostics_expose_selected_device_and_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot.hardware.sensors.usb_microphone import UsbMicrophone

    fake_sd = _FakeSoundDevice(
        devices=_devices(),
        default_input=1,
        actual_sample_rate=48_000,
        frames=[array("h", [4000] * 480)],
    )
    _install_sounddevice(monkeypatch, fake_sd)

    result = UsbMicrophone.diagnose_capture(
        input_device="default",
        sample_rate=16_000,
        channels=1,
        frame_ms=30,
        duration_s=0.05,
    )

    assert result["resolved_device_index"] == 1
    assert result["resolved_device_name"] == "Built-in Microphone"
    assert result["actual_stream_sample_rate"] == 48_000
    assert result["nonzero_audio"] is True
    assert result["rms_max"] > 0.0


# ---------------------------------------------------------------------------
# Stereo device handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stereo_device_channel_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the stream opens with a different channel count, it is rejected clearly."""
    from robot.hardware.sensors.usb_microphone import UsbMicrophone

    fake_sd = _FakeSoundDevice(
        devices=_devices(),
        default_input=1,
        actual_sample_rate=48_000,
        actual_channels=2,  # device returns stereo even though we asked for mono
    )
    _install_sounddevice(monkeypatch, fake_sd)

    mic = UsbMicrophone(input_device="default", _sample_rate_field=16_000, channels=1, frame_ms=30)
    with pytest.raises(RuntimeError, match="channel count mismatch"):
        mic._open_stream()


# ---------------------------------------------------------------------------
# Clean shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_stops_reader_thread_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing the microphone terminates the reader thread and stream."""
    from robot.hardware.sensors.usb_microphone import UsbMicrophone

    fake_sd = _FakeSoundDevice(
        devices=_devices(),
        default_input=1,
        frames=[array("h", [500] * 480)],
    )
    _install_sounddevice(monkeypatch, fake_sd)

    mic = UsbMicrophone(input_device="default", _sample_rate_field=16_000, channels=1, frame_ms=30)
    stream = mic.stream()
    await anext(stream)
    assert mic.thread_alive is True

    await mic.close()

    assert mic.thread_alive is False
    assert fake_sd.streams[0].stopped is True  # type: ignore[unreachable]
    assert fake_sd.streams[0]._closed is True


@pytest.mark.asyncio
async def test_close_drains_buffered_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """After close, the stream ends immediately even if chunks were buffered."""
    from robot.hardware.sensors.usb_microphone import UsbMicrophone

    fake_sd = _FakeSoundDevice(
        devices=_devices(),
        default_input=1,
        frames=[array("h", [300] * 480)],
    )
    _install_sounddevice(monkeypatch, fake_sd)

    mic = UsbMicrophone(input_device="default", _sample_rate_field=16_000, channels=1, frame_ms=30)
    stream = mic.stream()
    # Consume one chunk to start the reader thread.
    await anext(stream)
    # Give the reader thread time to buffer more chunks.
    await asyncio.sleep(0.05)
    await mic.close()
    # The stream must end immediately - no stale buffered chunks.
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


# ---------------------------------------------------------------------------
# Doctor CLI microphone diagnostic
# ---------------------------------------------------------------------------


def test_doctor_microphone_diagnostic_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The doctor CLI --microphone diagnostic can enumerate and capture."""
    import importlib

    doctor_mod = importlib.import_module("robot.cli.doctor")
    from robot.config import AppSettings, MicrophoneConfig

    fake_sd = _FakeSoundDevice(
        devices=_devices(),
        default_input=1,
        actual_sample_rate=48_000,
        frames=[array("h", [4000] * 480)],
    )
    _install_sounddevice(monkeypatch, fake_sd)

    settings = AppSettings(
        microphone=MicrophoneConfig(input_device="default", channels=1, sample_rate=16_000)
    )
    # Override env-var-driven defaults for a deterministic test.
    settings.microphone.channels = 1
    settings.microphone.sample_rate = 16_000
    # The diagnostic should run without raising.
    result = doctor_mod._run_microphone_diagnostic(settings, duration_s=0.05)
    assert result == 0  # nonzero_audio is True with 4000-amplitude samples


def test_doctor_microphone_diagnostic_reports_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    """The doctor CLI returns non-zero when only silence is captured."""
    import importlib

    doctor_mod = importlib.import_module("robot.cli.doctor")
    from robot.config import AppSettings, MicrophoneConfig

    fake_sd = _FakeSoundDevice(
        devices=_devices(),
        default_input=1,
        actual_sample_rate=48_000,
        frames=[array("h", [0] * 480)],  # silence
    )
    _install_sounddevice(monkeypatch, fake_sd)

    settings = AppSettings(
        microphone=MicrophoneConfig(input_device="default", channels=1, sample_rate=16_000)
    )
    # Override env-var-driven defaults for a deterministic test.
    settings.microphone.channels = 1
    settings.microphone.sample_rate = 16_000
    result = doctor_mod._run_microphone_diagnostic(settings, duration_s=0.05)
    assert result == 1  # nonzero_audio is False


# ---------------------------------------------------------------------------
# check_input_settings leniency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_input_settings_failure_does_not_block_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When check_input_settings rejects the rate, the stream still opens.

    On some ALSA/PipeWire setups check_input_settings rejects sample
    rates that PortAudio can actually resample at open time.  The
    pre-flight check must be advisory, not fatal.
    """
    from robot.hardware.sensors.usb_microphone import UsbMicrophone

    fake_sd = _FakeSoundDevice(
        devices=_devices(),
        default_input=1,
        actual_sample_rate=48_000,
        actual_channels=1,
        frames=[array("h", [1000] * 480)],
        check_error=RuntimeError("Unsupported sample rate"),
    )
    _install_sounddevice(monkeypatch, fake_sd)

    mic = UsbMicrophone(input_device="default", _sample_rate_field=16_000, channels=1, frame_ms=30)
    stream = mic.stream()
    chunk = await anext(stream)
    assert chunk.sample_rate == 48_000
    assert chunk.pcm != b""
    await mic.close()
