"""USB / default audio output driver backed by ``sounddevice``.

Plays audio through the system's audio output device.  Accepts an
:class:`AudioBuffer` whose format (sample rate, channels) may differ
from the device's native format; PortAudio handles the final
conversion.

Install with::

    uv pip install sounddevice

Configure with environment variables::

    DESKBOT_HARDWARE = real
    DESKBOT_AUDIO__OUTPUT_DEVICE = default
    DESKBOT_AUDIO__SAMPLE_RATE = 48000
"""

from __future__ import annotations

import asyncio
import struct
import threading
from dataclasses import dataclass, field

from robot.interfaces.audio import AudioBuffer
from robot.logging import get_logger

_log = get_logger("hardware.audio.usb")


def _s16le_to_float32(pcm: bytes) -> list[float]:
    """Convert s16le PCM bytes to float32 samples in [-1.0, 1.0]."""
    n_samples = len(pcm) // 2
    if n_samples == 0:
        return []
    samples = struct.unpack(f"<{n_samples}h", pcm)
    return [s / 32768.0 for s in samples]


def _float32_to_s16le(samples: list[float]) -> bytes:
    """Convert float32 samples in [-1.0, 1.0] back to s16le PCM bytes."""
    if not samples:
        return b""
    clamped = [max(-1.0, min(1.0, s)) for s in samples]
    return struct.pack(f"<{len(clamped)}h", *[int(c * 32767) for c in clamped])


@dataclass(slots=True)
class UsbSpeaker:
    """Real audio output using PortAudio via ``sounddevice``.

    The :attr:`sample_rate` and :attr:`channels` properties describe the
    speaker's *configured* output format.  Incoming :class:`AudioBuffer`
    instances may use any sample rate or channel count; the speaker uses
    ``sd.play`` with the buffer's own sample rate so PortAudio performs
    the correct resampling.
    """

    output_device: str | int = "default"
    _sample_rate: int = 48_000
    channels: int = 1
    latency: float = 0.1
    _sd: object | None = field(default=None, init=False, repr=False)
    _playing: bool = field(default=False, init=False)
    _stopped: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _device_index: int = field(default=0, init=False, repr=False)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def __post_init__(self) -> None:
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                f"sounddevice is required for UsbSpeaker: {exc!r}. "
                "Install with: uv pip install sounddevice"
            ) from exc
        self._sd = sd
        self._resolve_device()

    def _resolve_device(self) -> None:
        """Resolve the output device name/index to a sounddevice index."""
        import sounddevice as sd

        if isinstance(self.output_device, int):
            self._device_index: int = self.output_device
        elif self.output_device in ("default", ""):
            try:
                default_dev = sd.query_devices(kind="output")
                self._device_index = default_dev["index"]
            except Exception:
                self._device_index = sd.default.device[1]
        else:
            name = str(self.output_device).lower()
            found = False
            for i, dev in enumerate(sd.query_devices()):
                if name in dev["name"].lower() and dev["max_output_channels"] > 0:
                    self._device_index = i
                    found = True
                    break
            if not found:
                raise RuntimeError(f"no audio output device matching {self.output_device!r}")
        _log.info(
            "usb_speaker.resolved",
            device=self._device_index,
            requested=self.output_device,
        )

    async def play(self, buffer: AudioBuffer) -> None:
        """Play an :class:`AudioBuffer` and block until done.

        Converts s16le -> float32 for sounddevice and plays at the
        buffer's own sample rate so PortAudio performs correct
        resampling to the device's native rate.
        """
        if self._closed:
            raise RuntimeError("UsbSpeaker is closed")
        if buffer.is_empty:
            _log.debug("usb_speaker.play_empty")
            return

        import numpy as np

        # Convert s16le bytes -> float32 numpy array for sounddevice.
        n_samples = buffer.n_samples
        raw = struct.unpack(f"<{n_samples}h", buffer.pcm)
        float_data = np.array(raw, dtype=np.float32) / 32768.0

        if buffer.channels == 2 and len(float_data.shape) == 1:
            # Interleaved stereo: reshape to (n_frames, 2).
            float_data = float_data.reshape(-1, 2)

        _log.info(
            "audio.playback.started",
            sample_rate=buffer.sample_rate,
            channels=buffer.channels,
            sample_format=buffer.sample_format,
            bytes=len(buffer.pcm),
            device=self._device_index,
        )

        # Reset stop event for this playback.
        self._stop_event.clear()
        self._playing = True
        self._stopped = False

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def _play_blocking() -> None:
            import sounddevice as sd

            try:
                sd.play(
                    float_data,
                    samplerate=buffer.sample_rate,
                    device=self._device_index,
                    blocking=True,
                )
            except Exception as exc:
                _log.exception("audio.playback.failed")
                raise RuntimeError(f"UsbSpeaker playback failed: {exc}") from exc
            finally:
                self._playing = False

        if loop is not None:
            await loop.run_in_executor(None, _play_blocking)
        else:
            _play_blocking()

        _log.info(
            "audio.playback.completed",
            frames=buffer.n_frames,
            duration_s=round(buffer.duration_s, 3),
            sample_rate=buffer.sample_rate,
            channels=buffer.channels,
            device=self._device_index,
        )

    async def stop(self) -> None:
        """Interrupt whatever is currently playing."""
        import sounddevice as sd

        self._stopped = True
        self._playing = False
        self._stop_event.set()
        sd.stop()
        _log.info("usb_speaker.stopped")

    async def close(self) -> None:
        """Release the audio device."""
        if self._closed:
            return
        self._closed = True
        self._playing = False
        self._stop_event.set()
        import sounddevice as sd

        sd.stop()
        _log.info("usb_speaker.closed")


__all__ = ["UsbSpeaker", "_float32_to_s16le", "_s16le_to_float32"]
