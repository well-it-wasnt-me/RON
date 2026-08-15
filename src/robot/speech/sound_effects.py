"""Sound effects player for the DeskBot robot.

Loads WAV files from ``assets/sounds/`` and plays them through
the configured :class:`AudioOutput`. Supports random variation:
requesting ``"talk"`` will randomly pick from ``talk-1.wav``,
``talk-2.wav``, etc.

Configure with::

    DESKBOT_SOUNDS__ENABLED = true
"""

from __future__ import annotations

import io
import random
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path

from robot.events.bus import InMemoryEventBus
from robot.events.events import SoundEffectPlayed
from robot.interfaces.audio import AudioBuffer, AudioOutput, apply_volume
from robot.logging import get_logger

_log = get_logger("speech.sound_effects")

# Default directory for sound effect WAV files.
_DEFAULT_SOUNDS_DIR = Path("assets/sounds")


def _decode_sample(data: bytes, offset: int, sample_width: int) -> int:
    """Read one little-endian PCM sample and normalise it to signed 16-bit."""
    if sample_width == 1:
        return (data[offset] - 128) << 8
    if sample_width == 2:
        return int(struct.unpack_from("<h", data, offset)[0])
    if sample_width == 3:
        raw = int.from_bytes(data[offset : offset + 3], "little", signed=False)
        if raw & 0x800000:
            raw -= 1 << 24
        return raw >> 8
    if sample_width == 4:
        return int(struct.unpack_from("<i", data, offset)[0]) >> 16
    raise ValueError(f"unsupported WAV sample width: {sample_width}")


def _resample(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linearly resample mono signed-16 PCM without an optional dependency."""
    if source_rate == target_rate or not pcm:
        return pcm
    samples = list(struct.unpack(f"<{len(pcm) // 2}h", pcm))
    target_count = max(1, round(len(samples) * target_rate / source_rate))
    result: list[int] = []
    for index in range(target_count):
        source_index = index * source_rate / target_rate
        left = min(int(source_index), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        fraction = source_index - left
        result.append(round(samples[left] + (samples[right] - samples[left]) * fraction))
    return struct.pack(f"<{len(result)}h", *result)


def _apply_volume(pcm: bytes, volume: float) -> bytes:
    """Scale signed-16 PCM while preventing clipping."""
    if volume == 1.0 or not pcm:
        return pcm
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    scaled = [max(-32768, min(32767, round(sample * volume))) for sample in samples]
    return struct.pack(f"<{len(scaled)}h", *scaled)


def _wav_to_pcm(wav_bytes: bytes, target_sample_rate: int | None = None) -> tuple[bytes, int]:
    """Convert WAV bytes to raw s16le PCM and return (pcm, sample_rate).

    Handles stereo->mono downmix, common WAV sample widths, and optional
    conversion to an output device's sample rate. The returned PCM is always
    signed 16-bit little-endian mono, which is the :class:`AudioOutput`
    contract.
    """
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        pcm = wav.readframes(n_frames)

    if n_channels not in (1, 2):
        raise ValueError(f"unsupported WAV channel count: {n_channels}")

    frame_size = sample_width * n_channels
    mono_samples: list[int] = []
    for offset in range(0, len(pcm), frame_size):
        left = _decode_sample(pcm, offset, sample_width)
        if n_channels == 2:
            right = _decode_sample(pcm, offset + sample_width, sample_width)
            left = (left + right) // 2
        mono_samples.append(left)
    pcm = struct.pack(f"<{len(mono_samples)}h", *mono_samples)

    if target_sample_rate is not None and target_sample_rate != sample_rate:
        pcm = _resample(pcm, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return pcm, sample_rate


@dataclass(slots=True)
class SoundEffectsPlayer:
    """Play sound effects from ``assets/sounds/`` through an :class:`AudioOutput`.

    Parameters
    ----------
    sounds_dir:
        Directory containing WAV files. Defaults to ``assets/sounds/``.
    audio:
        The audio output device to play through. When ``None``, sounds
        are loaded but not played (useful for testing).
    enabled:
        Whether sound effects are enabled. When ``False``, :meth:`play`
        returns immediately without loading or playing anything.
    """

    sounds_dir: Path = field(default_factory=lambda: _DEFAULT_SOUNDS_DIR)
    audio: AudioOutput | None = None
    enabled: bool = True
    volume: float = 1.0
    bus: InMemoryEventBus | None = None
    _index: dict[str, list[Path]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.enabled:
            self._index_sounds()

    def _index_sounds(self) -> None:
        """Scan the sounds directory and group files by base name."""
        self._index.clear()
        if not self.sounds_dir.is_dir():
            _log.warning("sound_effects.dir_not_found", dir=str(self.sounds_dir))
            return
        for path in sorted(self.sounds_dir.glob("*.wav")):
            stem = path.stem
            parts = stem.split("__")
            name_part = parts[-1] if len(parts) >= 3 else stem
            base = name_part.rsplit("-", 1)[0] if name_part[-1].isdigit() else name_part
            # Assets are named ``small-robot-talk-1.wav``; expose their
            # useful semantic name as ``talk`` while retaining arbitrary
            # custom names unchanged.
            base = base.removeprefix("small-robot-")
            self._index.setdefault(base, []).append(path)

        _log.info(
            "sound_effects.indexed",
            sounds=list(self._index.keys()),
            total=sum(len(v) for v in self._index.values()),
        )

    async def play(self, name: str) -> bool:
        """Play a sound effect by name.

        If multiple WAV files match the name (e.g. ``talk-1.wav``,
        ``talk-2.wav``), one is picked at random.

        Returns ``True`` if a sound was played, ``False`` if not found
        or disabled.
        """
        if not self.enabled:
            return False

        paths = self._index.get(name, [])
        if not paths:
            _log.warning("sound_effects.not_found", name=name, available=list(self._index.keys()))
            return False

        path = random.choice(paths)
        _log.debug("sound_effects.playing", name=name, file=path.name)

        try:
            wav_bytes = path.read_bytes()
            target_sample_rate = self.audio.sample_rate if self.audio is not None else None
            pcm, sample_rate = _wav_to_pcm(wav_bytes, target_sample_rate)
            buffer = AudioBuffer(pcm=pcm, sample_rate=sample_rate, channels=1)
            if self.volume != 1.0 and not buffer.is_empty:
                buffer = AudioBuffer(
                    pcm=apply_volume(buffer.pcm, self.volume),
                    sample_rate=buffer.sample_rate,
                    channels=buffer.channels,
                    sample_format=buffer.sample_format,
                )
        except Exception:
            _log.exception("sound_effects.load_failed", file=str(path))
            return False

        if self.audio is not None and not buffer.is_empty:
            try:
                await self.audio.play(buffer)
            except Exception:
                _log.exception("sound_effects.play_failed", name=name)
                return False

        if self.bus is not None:
            await self.bus.publish(SoundEffectPlayed(name=name, filename=path.name))

        return True

    def list_sounds(self) -> list[str]:
        """Return the names of all available sound effects."""
        return sorted(self._index.keys())

    def has_sound(self, name: str) -> bool:
        """Check if a sound effect is available."""
        return name in self._index


__all__ = [
    "_DEFAULT_SOUNDS_DIR",
    "SoundEffectsPlayer",
    "_apply_volume",
    "_resample",
    "_wav_to_pcm",
]
