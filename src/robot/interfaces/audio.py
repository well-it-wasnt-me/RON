"""Audio output interface and explicit audio-format contract.

The :class:`AudioBuffer` dataclass is the central audio contract between
TTS providers (which *generate* audio) and :class:`AudioOutput`
implementations (which *play* audio).  Every buffer carries its own
sample rate, channel count, and sample format so the output layer never
has to guess.

Typical flow::

    TTS engine -> AudioBuffer(22050 Hz, mono, s16le)
                      |
    AudioOutput.play(buffer)
                      |
    format conversion (centralised, in this module)
                      |
    device-compatible PCM -> speaker / Bluetooth / USB

The output device is responsible for accepting audio in the format
described by the :class:`AudioBuffer` and converting it to whatever the
underlying hardware expects.  :func:`convert_audio` provides a
pure-Python (no external dependency) resampling and channel-conversion
utility that output implementations can call.
"""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# AudioBuffer - the explicit format contract
# ---------------------------------------------------------------------------

# Currently the only supported sample format.  The field exists so the
# contract is extensible if float32 or u8 support is needed later.
_S16LE = "s16le"


@dataclass(slots=True, frozen=True)
class AudioBuffer:
    """Audio data with explicit, provider-independent format metadata.

    Attributes
    ----------
    pcm:
        Raw PCM bytes.  Currently always signed 16-bit little-endian.
    sample_rate:
        Sample rate in Hz (e.g. 22050, 24000, 44100, 48000).
    channels:
        Number of channels (1 = mono, 2 = stereo).
    sample_format:
        PCM encoding.  Only ``"s16le"`` is currently supported.
    """

    pcm: bytes
    sample_rate: int
    channels: int = 1
    sample_format: str = _S16LE

    @property
    def sample_width(self) -> int:
        """Bytes per single (mono) sample."""
        if self.sample_format == _S16LE:
            return 2
        raise ValueError(f"unsupported sample format: {self.sample_format!r}")

    @property
    def n_frames(self) -> int:
        """Number of audio frames (one frame = one sample per channel)."""
        frame_bytes = self.sample_width * self.channels
        if frame_bytes == 0:
            return 0
        return len(self.pcm) // frame_bytes

    @property
    def n_samples(self) -> int:
        """Total interleaved samples across all channels."""
        return len(self.pcm) // self.sample_width

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        if self.sample_rate == 0:
            return 0.0
        return self.n_frames / self.sample_rate

    @property
    def is_empty(self) -> bool:
        return len(self.pcm) == 0

    # -- WAV convenience -------------------------------------------------

    def to_wav(self) -> bytes:
        """Wrap the PCM in a standard WAV container.

        Useful for passing to tools like ``paplay`` that can read WAV
        headers and determine the format automatically.
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(self.sample_width)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.pcm)
        return buf.getvalue()

    @staticmethod
    def from_wav(wav_bytes: bytes) -> AudioBuffer:
        """Parse a WAV byte string into an :class:`AudioBuffer`.

        Reads the actual sample rate, channel count, and sample width
        from the WAV header rather than assuming any defaults.
        """
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wav:
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            pcm = wav.readframes(n_frames)

        if sample_width != 2:
            raise ValueError(
                f"AudioBuffer.from_wav only supports 16-bit WAV, got sample_width={sample_width}"
            )

        return AudioBuffer(
            pcm=pcm,
            sample_rate=sample_rate,
            channels=n_channels,
            sample_format=_S16LE,
        )

    @staticmethod
    def silence(duration_s: float, sample_rate: int = 22050, channels: int = 1) -> AudioBuffer:
        """Create a silent AudioBuffer of the given duration."""
        n_frames = int(sample_rate * duration_s)
        pcm = b"\x00\x00" * n_frames * channels
        return AudioBuffer(pcm=pcm, sample_rate=sample_rate, channels=channels)


# ---------------------------------------------------------------------------
# Centralised format conversion
# ---------------------------------------------------------------------------


def resample(pcm: bytes, source_rate: int, target_rate: int, channels: int = 1) -> bytes:
    """Linearly resample s16le PCM from *source_rate* to *target_rate*.

    Handles mono and interleaved multi-channel data.  Pure Python (no
    numpy required).  Uses linear interpolation between samples.
    """
    if source_rate == target_rate or not pcm:
        return pcm

    sample_width = 2
    frame_size = sample_width * channels
    n_source_frames = len(pcm) // frame_size
    if n_source_frames == 0:
        return b""

    n_target_frames = max(1, round(n_source_frames * target_rate / source_rate))

    # Unpack all source samples as a flat list of interleaved channel values.
    n_source_samples = n_source_frames * channels
    source_samples = struct.unpack(f"<{n_source_samples}h", pcm[: n_source_samples * 2])

    result_samples: list[int] = []
    for ti in range(n_target_frames):
        src_idx = ti * source_rate / target_rate
        left = min(int(src_idx), n_source_frames - 1)
        right = min(left + 1, n_source_frames - 1)
        frac = src_idx - left

        for ch in range(channels):
            s_left = source_samples[left * channels + ch]
            s_right = source_samples[right * channels + ch]
            interpolated = round(s_left + (s_right - s_left) * frac)
            result_samples.append(max(-32768, min(32767, interpolated)))

    return struct.pack(f"<{len(result_samples)}h", *result_samples)


def convert_channels(pcm: bytes, source_channels: int, target_channels: int) -> bytes:
    """Convert s16le PCM between channel counts.

    Supports mono <-> stereo.  Mono to stereo duplicates the channel;
    stereo to mono averages left and right.
    """
    if source_channels == target_channels or not pcm:
        return pcm

    sample_width = 2
    n_frames = len(pcm) // (sample_width * source_channels)
    if n_frames == 0:
        return b""

    source_samples = struct.unpack(
        f"<{n_frames * source_channels}h", pcm[: n_frames * source_channels * 2]
    )
    result_samples: list[int] = []

    if source_channels == 1 and target_channels == 2:
        for s in source_samples:
            result_samples.append(s)
            result_samples.append(s)
    elif source_channels == 2 and target_channels == 1:
        for i in range(0, len(source_samples), 2):
            left = source_samples[i]
            right = source_samples[i + 1] if i + 1 < len(source_samples) else left
            result_samples.append((left + right) // 2)
    else:
        raise ValueError(f"unsupported channel conversion: {source_channels} -> {target_channels}")

    return struct.pack(f"<{len(result_samples)}h", *result_samples)


def convert_audio(
    buffer: AudioBuffer,
    target_sample_rate: int,
    target_channels: int,
) -> AudioBuffer:
    """Convert an :class:`AudioBuffer` to the target format.

    Performs sample-rate conversion and channel conversion in sequence.
    Returns a new :class:`AudioBuffer` with the converted PCM.  When no
    conversion is needed, the original buffer is returned unchanged.
    """
    if buffer.sample_rate == target_sample_rate and buffer.channels == target_channels:
        return buffer

    pcm = buffer.pcm
    if buffer.channels != target_channels:
        pcm = convert_channels(pcm, buffer.channels, target_channels)
    if buffer.sample_rate != target_sample_rate:
        pcm = resample(pcm, buffer.sample_rate, target_sample_rate, target_channels)

    return AudioBuffer(
        pcm=pcm,
        sample_rate=target_sample_rate,
        channels=target_channels,
        sample_format=buffer.sample_format,
    )


def apply_volume(pcm: bytes, volume: float) -> bytes:
    """Scale s16le PCM by *volume* (0.0 - 1.0) while preventing clipping."""
    if volume == 1.0 or not pcm:
        return pcm
    n_samples = len(pcm) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm)
    scaled = [max(-32768, min(32767, round(s * volume))) for s in samples]
    return struct.pack(f"<{n_samples}h", *scaled)


# ---------------------------------------------------------------------------
# AudioOutput protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AudioOutput(Protocol):
    """Speaker / DAC interface.

    Implementations accept an :class:`AudioBuffer` whose ``sample_rate``,
    ``channels``, and ``sample_format`` describe the actual audio data.
    The output implementation is responsible for converting to whatever
    its underlying hardware requires (see :func:`convert_audio`).
    """

    @property
    def sample_rate(self) -> int:
        """Native sample rate of the output device (Hz)."""
        ...

    @property
    def channels(self) -> int:
        """Number of output channels (1 = mono, 2 = stereo)."""
        ...

    async def play(self, buffer: AudioBuffer) -> None:
        """Play audio described by *buffer* and block until done."""
        ...

    async def stop(self) -> None:
        """Interrupt whatever is currently playing."""
        ...

    async def close(self) -> None:
        """Release the audio device."""
        ...


__all__ = [
    "AudioBuffer",
    "AudioOutput",
    "apply_volume",
    "convert_audio",
    "convert_channels",
    "resample",
]
