"""Fake audio output."""

from __future__ import annotations

from robot.interfaces.audio import AudioBuffer


class FakeAudioOutput:
    def __init__(self, sample_rate: int = 48_000, channels: int = 1) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self.played: list[AudioBuffer] = []
        self.stopped = False
        self.closed = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def play(self, buffer: AudioBuffer) -> None:
        self.played.append(buffer)

    async def stop(self) -> None:
        self.stopped = True

    async def close(self) -> None:
        self.closed = True
