"""Speech-to-text interface and mock implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from robot.interfaces.microphone import AudioChunk
from robot.logging import get_logger

_log = get_logger("speech.stt")


@runtime_checkable
class SpeechToText(Protocol):
    async def transcribe(self, audio: AudioChunk | AsyncIterator[AudioChunk]) -> str:
        """Transcribe audio to text."""


class MockSTT:
    """Returns a fixed string for every :meth:`transcribe` call."""

    def __init__(self, transcript: str = "hello deskbot") -> None:
        self._transcript = transcript
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    async def transcribe(self, audio: AudioChunk | AsyncIterator[AudioChunk]) -> str:
        self._calls += 1
        _log.debug("stt.transcribe", calls=self._calls)
        return self._transcript

    async def close(self) -> None:
        return None


__all__ = ["MockSTT", "SpeechToText"]
