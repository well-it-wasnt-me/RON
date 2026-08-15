"""Mock microphone returning a deterministic stream of silence."""

from __future__ import annotations

import asyncio
import struct
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from robot.interfaces.microphone import AudioChunk
from robot.logging import get_logger

_log = get_logger("hardware.sensors.microphone.mock")


@dataclass(slots=True)
class MockMicrophone:
    """Microphone that emits silence at a fixed sample rate."""

    sample_rate: int = 16_000
    channels: int = 1
    frame_ms: int = 30
    _wav_path: Path | None = None
    _stopped: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)

    def use_wav(self, path: Path) -> None:
        self._wav_path = path

    def stream(self) -> AsyncIterator[AudioChunk]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[AudioChunk]:
        if self._closed:
            raise RuntimeError("microphone is closed")
        frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        chunk = struct.pack(f"<{frame_samples}h", *([0] * frame_samples))
        idx = 0
        while not self._stopped:
            await asyncio.sleep(self.frame_ms / 1000)
            yield AudioChunk(
                pcm=chunk,
                sample_rate=self.sample_rate,
                channels=self.channels,
                timestamp=idx * self.frame_ms / 1000,
            )
            idx += 1

    async def close(self) -> None:
        self._closed = True
        self._stopped = True

    def stop(self) -> None:
        self._stopped = True


def load_wav_as_chunks(path: Path, frame_ms: int = 30) -> list[AudioChunk]:
    """Read a WAV file from disk and return :class:`AudioChunk` instances."""
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        data = wav.readframes(wav.getnframes())
    frame_bytes = int(sample_rate * frame_ms / 1000) * channels * sample_width
    chunks: list[AudioChunk] = []
    for i in range(0, len(data), frame_bytes):
        chunks.append(
            AudioChunk(
                pcm=data[i : i + frame_bytes],
                sample_rate=sample_rate,
                channels=channels,
                timestamp=i / (sample_rate * channels * sample_width),
            )
        )
    return chunks


def write_wav(target: BinaryIO | Path, sample_rate: int, pcm: bytes) -> None:
    """Write a mono 16-bit WAV file (test helper)."""
    if isinstance(target, Path):
        target = target.open("wb")
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


__all__ = ["MockMicrophone", "load_wav_as_chunks", "write_wav"]
