"""Mock audio output that records what was played."""

from __future__ import annotations

from robot.interfaces.audio import AudioBuffer
from robot.logging import get_logger

_log = get_logger("hardware.audio.mock")


class MockAudioOutput:
    """In-memory :class:`AudioOutput` for tests.

    Records every :class:`AudioBuffer` passed to :meth:`play` so tests
    can inspect the exact format (sample rate, channels, PCM) that was
    sent.
    """

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
        if self.closed:
            raise RuntimeError("audio output is closed")
        self.played.append(buffer)
        _log.info(
            "audio.mock_play",
            bytes=len(buffer.pcm),
            sample_rate=buffer.sample_rate,
            channels=buffer.channels,
            msg="audio fell back to mock - nothing will be heard",
        )

    async def stop(self) -> None:
        self.stopped = True

    async def close(self) -> None:
        self.closed = True


__all__ = ["MockAudioOutput"]
