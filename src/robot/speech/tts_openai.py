"""OpenAI text-to-speech.

Wraps the ``/v1/audio/speech`` endpoint. Works against any
OpenAI-compatible TTS (OpenAI's ``tts-1``/``tts-1-hd``, LocalAI,
vLLM with the TTS extension, etc.).

Install with::

    uv pip install openai httpx

The OpenAI TTS API returns raw PCM (``response_format="pcm"``) at
**24000 Hz, mono, s16le**.  This format is propagated through the
returned :class:`AudioBuffer`.
"""

from __future__ import annotations

import httpx

from robot.interfaces.audio import AudioBuffer, AudioOutput
from robot.logging import get_logger

_log = get_logger("speech.tts.openai")

# OpenAI's "pcm" response format is always 24 kHz mono s16le.
_OPENAI_PCM_SAMPLE_RATE = 24_000


class OpenAITTS:
    """OpenAI TTS.

    Parameters
    ----------
    api_key:
        Bearer token. Empty for local servers.
    base_url:
        API base URL.
    model:
        Model name (e.g. ``"tts-1"``).
    voice:
        Voice name (``"alloy"``, ``"echo"``, ``"fable"``, ``"onyx"``,
        ``"nova"``, ``"shimmer"`` for OpenAI-hosted).
    audio:
        Optional :class:`AudioOutput` for playing the synthesised audio.
    timeout_s:
        HTTP request timeout.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "tts-1",
        voice: str = "alloy",
        audio: AudioOutput | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._voice = voice
        self._audio = audio
        self._timeout_s = timeout_s

    async def speak(self, text: str) -> AudioBuffer:
        """Synthesise *text* and return an :class:`AudioBuffer`.

        The OpenAI ``pcm`` response format is 24 kHz mono s16le.
        """
        url = f"{self._base_url}/audio/speech"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "voice": self._voice,
            "input": text,
            "response_format": "pcm",
        }
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        buffer = AudioBuffer(
            pcm=response.content,
            sample_rate=_OPENAI_PCM_SAMPLE_RATE,
            channels=1,
        )
        _log.debug(
            "openai_tts.synthesized",
            pcm_len=len(buffer.pcm),
            sample_rate=buffer.sample_rate,
            text=text[:80],
        )
        return buffer

    async def close(self) -> None:
        return None


__all__ = ["OpenAITTS"]
