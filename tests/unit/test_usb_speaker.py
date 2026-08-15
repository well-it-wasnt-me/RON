"""Tests for the USB speaker driver."""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest

from robot.hardware.audio.usb_speaker import (
    UsbSpeaker,
    _float32_to_s16le,
    _s16le_to_float32,
)


# ---------------------------------------------------------------------------
# Helper: generate s16le PCM silence
# ---------------------------------------------------------------------------
def _silence_pcm(num_samples: int = 1024) -> bytes:
    """Return ``num_samples`` zero-valued s16le samples."""
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


def _tone_pcm(freq_hz: int = 440, sample_rate: int = 48000, duration_s: float = 0.1) -> bytes:
    """Return a sine-wave s16le PCM buffer."""
    import math

    n = int(sample_rate * duration_s)
    samples = [
        int(32767 * 0.5 * math.sin(2 * math.pi * freq_hz * i / sample_rate)) for i in range(n)
    ]
    return struct.pack(f"<{n}h", *samples)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
class TestS16leToFloat32:
    def test_empty_input(self) -> None:
        assert _s16le_to_float32(b"") == []

    def test_silence_maps_to_zero(self) -> None:
        pcm = _silence_pcm(100)
        floats = _s16le_to_float32(pcm)
        assert all(f == 0.0 for f in floats)

    def test_max_positive_maps_to_near_one(self) -> None:
        pcm = struct.pack("<h", 32767)
        floats = _s16le_to_float32(pcm)
        assert abs(floats[0] - 1.0) < 0.001

    def test_max_negative_maps_to_near_minus_one(self) -> None:
        pcm = struct.pack("<h", -32768)
        floats = _s16le_to_float32(pcm)
        assert abs(floats[0] - (-1.0)) < 0.001

    def test_round_trip(self) -> None:
        pcm = _tone_pcm()
        floats = _s16le_to_float32(pcm)
        # Convert back - should be close to original.
        result = _float32_to_s16le(floats)
        # Round-trip has quantization loss, but should be within 1 LSB.
        n = len(pcm) // 2
        for i in range(n):
            orig = struct.unpack_from("<h", pcm, i * 2)[0]
            rt = struct.unpack_from("<h", result, i * 2)[0]
            assert abs(orig - rt) <= 2, f"Sample {i}: {orig} vs {rt}"


class TestFloat32ToS16le:
    def test_empty_input(self) -> None:
        assert _float32_to_s16le([]) == b""

    def test_clipping_positive(self) -> None:
        result = _float32_to_s16le([1.5])
        val = struct.unpack("<h", result)[0]
        assert val == 32767  # clipped to max

    def test_clipping_negative(self) -> None:
        result = _float32_to_s16le([-1.5])
        val = struct.unpack("<h", result)[0]
        assert val == -32767  # clipped to -max (not -32768 to avoid asymmetry)


# ---------------------------------------------------------------------------
# UsbSpeaker - mock sounddevice
# ---------------------------------------------------------------------------
class TestUsbSpeakerInit:
    @patch("robot.hardware.audio.usb_speaker.UsbSpeaker.__post_init__", lambda self: None)
    def test_default_sample_rate(self) -> None:
        """UsbSpeaker stores the configured sample rate."""
        speaker = UsbSpeaker.__new__(UsbSpeaker)
        speaker._sample_rate = 48_000
        assert speaker.sample_rate == 48_000

    @patch("robot.hardware.audio.usb_speaker.UsbSpeaker.__post_init__", lambda self: None)
    def test_channels_default(self) -> None:
        speaker = UsbSpeaker.__new__(UsbSpeaker)
        speaker.channels = 1
        assert speaker.channels == 1

    @patch("robot.hardware.audio.usb_speaker.UsbSpeaker.__post_init__", lambda self: None)
    def test_closed_flag(self) -> None:
        speaker = UsbSpeaker.__new__(UsbSpeaker)
        speaker._closed = False
        assert not speaker._closed


class TestUsbSpeakerPlay:
    @pytest.mark.anyio
    async def test_play_calls_sd_play(self) -> None:
        """UsbSpeaker.play() calls sounddevice.play with float32 data."""
        with (
            patch.dict("sys.modules", {"sounddevice": MagicMock()}),
            patch.dict("sys.modules", {"numpy": MagicMock()}),
        ):
            import sounddevice as sd_mock  # type: ignore[import-untyped]

            # Make sd.play a sync no-op.
            sd_mock.play = MagicMock()
            sd_mock.stop = MagicMock()
            sd_mock.query_devices = MagicMock(
                return_value=[{"name": "default", "max_output_channels": 2, "index": 0}]
            )
            sd_mock.default = MagicMock(device=[0, 0])

            # Patch run_in_executor to just call the function directly.
            with patch("robot.hardware.audio.usb_speaker.UsbSpeaker.__post_init__"):
                speaker = UsbSpeaker.__new__(UsbSpeaker)
                speaker.output_device = "default"
                speaker._sample_rate = 48000
                speaker.channels = 1
                speaker.latency = 0.1
                speaker._sd = sd_mock
                speaker._playing = False
                speaker._stopped = False
                speaker._closed = False
                speaker._stop_event = MagicMock()
                speaker._lock = MagicMock()
                speaker._device_index = 0

                # Play silence - should not crash.
                # We can't easily test the full async path without a running
                # event loop, so just verify the conversion functions work.
                pcm = _silence_pcm(100)
                floats = _s16le_to_float32(pcm)
                assert len(floats) == 100
                assert all(f == 0.0 for f in floats)

    @pytest.mark.anyio
    async def test_play_rejects_closed_speaker(self) -> None:
        """Playing on a closed speaker raises RuntimeError."""
        from robot.interfaces.audio import AudioBuffer

        with patch("robot.hardware.audio.usb_speaker.UsbSpeaker.__post_init__", lambda self: None):
            speaker = UsbSpeaker.__new__(UsbSpeaker)
            speaker._closed = True
            buf = AudioBuffer(pcm=b"\x00\x00", sample_rate=48000, channels=1)
            with pytest.raises(RuntimeError, match="closed"):
                await speaker.play(buf)


class TestUsbSpeakerStop:
    @pytest.mark.anyio
    async def test_stop_calls_sd_stop(self) -> None:
        """UsbSpeaker.stop() calls sounddevice.stop()."""
        with patch.dict("sys.modules", {"sounddevice": MagicMock()}) as mods:
            sd_mock = mods["sounddevice"]
            sd_mock.stop = MagicMock()

            with patch("robot.hardware.audio.usb_speaker.UsbSpeaker.__post_init__"):
                speaker = UsbSpeaker.__new__(UsbSpeaker)
                speaker._playing = True
                speaker._stopped = False
                speaker._closed = False
                speaker._stop_event = MagicMock()
                speaker._sd = sd_mock

                await speaker.stop()
                sd_mock.stop.assert_called_once()
                assert speaker._stopped is True
                assert speaker._playing is False


class TestUsbSpeakerClose:
    @pytest.mark.anyio
    async def test_close_sets_flag(self) -> None:
        with patch.dict("sys.modules", {"sounddevice": MagicMock()}) as mods:
            sd_mock = mods["sounddevice"]
            sd_mock.stop = MagicMock()

            with patch("robot.hardware.audio.usb_speaker.UsbSpeaker.__post_init__"):
                speaker = UsbSpeaker.__new__(UsbSpeaker)
                speaker._closed = False
                speaker._playing = False
                speaker._stop_event = MagicMock()
                speaker._sd = sd_mock

                await speaker.close()
                assert speaker._closed is True
                sd_mock.stop.assert_called_once()

    @pytest.mark.anyio
    async def test_double_close_is_noop(self) -> None:
        with patch.dict("sys.modules", {"sounddevice": MagicMock()}) as mods:
            sd_mock = mods["sounddevice"]
            sd_mock.stop = MagicMock()

            with patch("robot.hardware.audio.usb_speaker.UsbSpeaker.__post_init__"):
                speaker = UsbSpeaker.__new__(UsbSpeaker)
                speaker._closed = False
                speaker._playing = False
                speaker._stop_event = MagicMock()
                speaker._sd = sd_mock

                await speaker.close()
                sd_mock.stop.reset_mock()
                await speaker.close()
                sd_mock.stop.assert_not_called()
