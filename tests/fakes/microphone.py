"""Fake microphone that yields a scripted stream of audio chunks."""

from __future__ import annotations

from dataclasses import dataclass, field

from robot.interfaces.microphone import AudioChunk


@dataclass
class FakeMicrophone:
    sample_rate: int = 16_000
    channels: int = 1
    frame_ms: int = 30
    chunks: list[AudioChunk] = field(default_factory=list)
    closed: bool = False

    async def stream(self):
        for chunk in self.chunks:
            yield chunk

    async def close(self) -> None:
        self.closed = True
