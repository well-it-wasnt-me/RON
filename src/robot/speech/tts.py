"""Text-to-speech interface and mock implementation.

The :class:`TextToSpeech` protocol returns an :class:`AudioBuffer` that
carries the actual sample rate, channel count, and sample format of the
synthesised audio.  This allows any :class:`AudioOutput` to play the
audio correctly without guessing the format.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from robot.interfaces.audio import AudioBuffer, AudioOutput
from robot.logging import get_logger

_log = get_logger("speech.tts")


@runtime_checkable
class TextToSpeech(Protocol):
    async def speak(self, text: str) -> AudioBuffer:
        """Synthesise speech for *text* and return an :class:`AudioBuffer`.

        The returned buffer carries the actual sample rate, channels,
        and sample format of the synthesised audio.
        """

    async def close(self) -> None: ...


class MockTTS:
    """Returns an empty :class:`AudioBuffer` but records every spoken text.

    The optional *audio* parameter is accepted for backward
    compatibility but is **not** used for playback.  :class:`MockTTS`
    never produces physical speech; the caller is responsible for
    detecting the mock and reporting degradation.
    """

    def __init__(self, audio: AudioOutput | None = None) -> None:
        self._audio = audio
        self._spoken: list[str] = []

    @property
    def spoken(self) -> list[str]:
        return list(self._spoken)

    async def speak(self, text: str) -> AudioBuffer:
        self._spoken.append(text)
        _log.debug("tts.speak", text=text)
        return AudioBuffer(pcm=b"", sample_rate=22050, channels=1)

    async def close(self) -> None:
        return None


__all__ = ["MockTTS", "TextToSpeech"]
