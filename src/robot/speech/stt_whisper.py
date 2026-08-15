"""Speech-to-text using OpenAI's Whisper (local whisper.cpp / faster-whisper
or hosted OpenAI ``whisper-1`` endpoint).

Uses the ``/v1/audio/transcriptions`` endpoint, which accepts a WAV blob
and returns the transcript text. Works against any OpenAI-compatible
speech endpoint (whisper.cpp server, faster-whisper-server, etc.).

Install with::

    uv pip install openai httpx
"""

from __future__ import annotations

import io
import wave
from collections.abc import AsyncIterator

import httpx

from robot.interfaces.microphone import AudioChunk
from robot.logging import get_logger

_log = get_logger("speech.stt.whisper")


class WhisperSTT:
    """OpenAI Whisper-compatible STT.

    Parameters
    ----------
    api_key:
        Bearer token. Empty for local servers.
    base_url:
        API base URL.
    model:
        Whisper model name (e.g. ``"whisper-1"`` for hosted, ``"large-v3"`` /
        ``"small"`` for local whisper.cpp).
    language:
        Two-letter language code or ``None`` for auto-detect.
    timeout_s:
        HTTP request timeout.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "whisper-1",
        language: str | None = "en",
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._language = language
        self._timeout_s = timeout_s

    async def transcribe(self, audio: AudioChunk) -> str:
        # The OpenAI Whisper endpoint expects a multipart/form-data POST
        # with a `file` field containing a complete WAV file.
        wav_bytes = _pcm_to_wav(audio)
        url = f"{self._base_url}/audio/transcriptions"
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = {
            "model": (None, self._model),
            "response_format": (None, "json"),
        }
        if self._language:
            data["language"] = (None, self._language)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        result = response.json()
        return str(result.get("text", ""))

    async def close(self) -> None:
        return None


def _pcm_to_wav(chunk: AudioChunk) -> bytes:
    """Wrap an :class:`AudioChunk` into a complete mono WAV blob."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(chunk.channels)
        wav.setsampwidth(2)  # int16
        wav.setframerate(chunk.sample_rate)
        wav.writeframes(chunk.pcm)
    return buf.getvalue()


class StreamingWhisperAdapter:
    """Adapter that makes :class:`WhisperSTT` work with the
    :class:`SpeechToText` protocol's ``transcribe(audio)`` method
    which accepts either a single :class:`AudioChunk` or an
    async iterator of chunks.

    The original :class:`WhisperSTT` only accepts a single
    :class:`AudioChunk`. This adapter concatenates all chunks from
    an async iterator into one WAV file before calling Whisper.
    """

    def __init__(self, inner: WhisperSTT) -> None:
        self._inner = inner

    async def transcribe(self, audio: AudioChunk | AsyncIterator[AudioChunk]) -> str:
        """Transcribe audio, handling both single chunks and streams."""

        import io as _io
        import wave as _wave

        chunks: list[bytes] = []
        sr = 16_000
        ch = 1
        if isinstance(audio, AudioChunk):
            chunks.append(audio.pcm)
            sr = audio.sample_rate
            ch = audio.channels
        else:
            async for c in audio:
                chunks.append(c.pcm)
                sr = c.sample_rate
                ch = c.channels
        buf = _io.BytesIO()
        with _wave.open(buf, "wb") as wav:
            wav.setnchannels(ch)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(b"".join(chunks))
        wav_bytes = buf.getvalue()
        # Use the WhisperSTT API directly via httpx
        import httpx as _httpx

        url = self._inner._base_url + "/audio/transcriptions"
        headers = {}
        if self._inner._api_key:
            headers["Authorization"] = f"Bearer {self._inner._api_key}"
        data = {"model": (None, self._inner._model), "response_format": (None, "json")}
        if self._inner._language:
            data["language"] = (None, self._inner._language)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        async with _httpx.AsyncClient(timeout=self._inner._timeout_s) as client:
            response = await client.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        return str(response.json().get("text", ""))

    async def close(self) -> None:
        return None


__all__ = ["StreamingWhisperAdapter", "WhisperSTT"]
