"""ElevenLabs text-to-speech.

Wraps the ``/v1/text-to-speech/{voice_id}`` endpoint. ElevenLabs provides
high-quality cloud-based TTS with many natural-sounding voices.

Install with::

    uv pip install httpx

Configure with::

    DESKBOT_TTS__PROVIDER=elevenlabs
    DESKBOT_TTS__ELEVENLABS__API_KEY=your-api-key
    DESKBOT_TTS__ELEVENLABS__VOICE_ID=21m00Tcm4TlvDq8ikWAM

The ``output_format`` parameter determines the PCM format.  For
``pcm_*`` formats the sample rate is encoded in the format name (e.g.
``pcm_16000`` = 16000 Hz).  This is parsed and propagated through the
returned :class:`AudioBuffer`.
"""

from __future__ import annotations

import httpx

from robot.errors import ConfigurationError
from robot.interfaces.audio import AudioBuffer, AudioOutput
from robot.logging import get_logger

_log = get_logger("speech.tts.elevenlabs")

_BASE_URL = "https://api.elevenlabs.io"

# Map ElevenLabs output_format names to sample rates.
# See https://elevenlabs.io/docs/api-reference/text-to-speech
_ELEVENLABS_SAMPLE_RATES: dict[str, int] = {
    "pcm_8000": 8000,
    "pcm_16000": 16000,
    "pcm_22050": 22050,
    "pcm_24000": 24000,
    "pcm_44100": 44100,
    "pcm_48000": 48000,
}


class ElevenLabsTTS:
    """Cloud-based TTS using the `ElevenLabs <https://elevenlabs.io>`_ API.

    The ``output_format`` parameter determines the sample rate of the
    returned PCM.  For ``pcm_*`` formats the rate is parsed from the
    format name and propagated through the :class:`AudioBuffer`.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "pcm_16000",
        audio: AudioOutput | None = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        timeout_s: float = 30.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "ElevenLabs API key is required. "
                "Set DESKBOT_TTS__ELEVENLABS__API_KEY or pass api_key=."
            )
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._audio = audio
        self._stability = stability
        self._similarity_boost = similarity_boost
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return f"elevenlabs:{self._voice_id}"

    def _sample_rate_for_format(self) -> int:
        """Determine the sample rate from the output_format string."""
        rate = _ELEVENLABS_SAMPLE_RATES.get(self._output_format)
        if rate is not None:
            return rate
        # Default for unknown pcm formats or non-pcm formats.
        if self._output_format.startswith("pcm_"):
            try:
                return int(self._output_format.removeprefix("pcm_"))
            except ValueError:
                pass
        return 16000

    async def speak(self, text: str) -> AudioBuffer:
        """Synthesise *text* and return an :class:`AudioBuffer`.

        Calls the ElevenLabs ``/v1/text-to-speech/{voice_id}`` endpoint
        and returns the audio with the correct sample rate derived from
        the ``output_format``.
        """
        url = f"{_BASE_URL}/v1/text-to-speech/{self._voice_id}"
        headers = {
            "Content-Type": "application/json",
            "xi-api-key": self._api_key,
        }
        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {
                "stability": self._stability,
                "similarity_boost": self._similarity_boost,
            },
        }
        params = {"output_format": self._output_format}

        _log.debug(
            "elevenlabs.speak",
            text=text[:80],
            voice_id=self._voice_id,
            model_id=self._model_id,
            output_format=self._output_format,
        )

        sample_rate = self._sample_rate_for_format()

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(url, json=payload, headers=headers, params=params)
        except httpx.TimeoutException:
            _log.error("elevenlabs.timeout", text=text[:80])
            return AudioBuffer(pcm=b"", sample_rate=sample_rate, channels=1)
        except httpx.ConnectError as exc:
            _log.error("elevenlabs.connection_failed", error=str(exc))
            return AudioBuffer(pcm=b"", sample_rate=sample_rate, channels=1)

        if response.status_code == 401:
            _log.error("elevenlabs.unauthorized", hint="Check your API key")
            raise ConfigurationError("ElevenLabs API key is invalid or unauthorized")
        if response.status_code == 429:
            _log.error("elevenlabs.rate_limited")
            return AudioBuffer(pcm=b"", sample_rate=sample_rate, channels=1)
        if response.status_code >= 400:
            _log.error(
                "elevenlabs.api_error",
                status=response.status_code,
                body=response.text[:200],
            )
            return AudioBuffer(pcm=b"", sample_rate=sample_rate, channels=1)

        buffer = AudioBuffer(
            pcm=response.content,
            sample_rate=sample_rate,
            channels=1,
        )
        _log.debug(
            "elevenlabs.synthesized",
            pcm_len=len(buffer.pcm),
            sample_rate=buffer.sample_rate,
            text=text[:80],
        )

        return buffer

    async def close(self) -> None:
        """No persistent resources to release."""
        _log.info("elevenlabs.closed")


__all__ = ["ElevenLabsTTS"]
