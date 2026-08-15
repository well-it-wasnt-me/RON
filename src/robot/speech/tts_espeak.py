"""eSpeak-NG TTS - ultra-lightweight local speech synthesis.

eSpeak-NG is a compact, open-source speech synthesizer that works
entirely offline and runs well on a Raspberry Pi. It produces
lower-quality audio than Piper or cloud TTS, but it has virtually
no latency and no model download requirement.

Install eSpeak-NG on the Pi::

    sudo apt install espeak-ng

Configure with::

    DESKBOT_TTS__PROVIDER = espeak
"""

from __future__ import annotations

import asyncio
import shutil

from robot.interfaces.audio import AudioBuffer, AudioOutput
from robot.logging import get_logger

_log = get_logger("speech.tts.espeak")


class EspeakNGTTS:
    """Local text-to-speech using `eSpeak-NG <https://github.com/espeak-ng/espeak-ng>`_.

    The synthesised audio format is read from the WAV header produced by
    ``espeak-ng --stdout`` and propagated faithfully through an
    :class:`AudioBuffer`.  The output layer is responsible for any
    resampling or channel conversion needed.
    """

    def __init__(
        self,
        voice: str = "en",
        speed: int = 175,
        pitch: int = 50,
        sample_rate: int = 22050,
        audio: AudioOutput | None = None,
    ) -> None:
        self._voice = voice
        self._speed = speed
        self._pitch = pitch
        self._sample_rate = sample_rate
        self._audio = audio
        self._espeak_bin: str | None = None

    @property
    def name(self) -> str:
        return f"espeak-ng:{self._voice}"

    def _find_espeak(self) -> str:
        """Locate the ``espeak-ng`` binary."""
        if self._espeak_bin is not None:
            return self._espeak_bin
        path = shutil.which("espeak-ng")
        if path is None:
            path = shutil.which("espeak")
        if path is None:
            raise FileNotFoundError(
                "eSpeak-NG not found. Install it with: sudo apt install espeak-ng"
            )
        self._espeak_bin = path
        return path

    async def speak(self, text: str) -> AudioBuffer:
        """Synthesise *text* and return an :class:`AudioBuffer`.

        Uses ``espeak-ng --stdout`` to produce WAV audio, then parses the
        WAV header to extract the actual sample rate and channel count.
        """
        espeak = self._find_espeak()
        cmd = [
            espeak,
            "-v",
            self._voice,
            "-s",
            str(self._speed),
            "-p",
            str(self._pitch),
            "--stdout",
            text,
        ]
        _log.debug("espeak.speak", text=text[:80], voice=self._voice)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            _log.error(
                "espeak.failed",
                returncode=process.returncode,
                stderr=stderr.decode(errors="replace")[:200],
            )
            return AudioBuffer(pcm=b"", sample_rate=self._sample_rate, channels=1)

        if not stdout:
            _log.warning("espeak.empty_output", text=text[:80])
            return AudioBuffer(pcm=b"", sample_rate=self._sample_rate, channels=1)

        # Parse the WAV header to extract the actual format.
        buffer = self._parse_wav(stdout)
        _log.debug(
            "espeak.synthesized",
            pcm_len=len(buffer.pcm),
            sample_rate=buffer.sample_rate,
            channels=buffer.channels,
            text=text[:80],
        )

        return buffer

    def _parse_wav(self, wav_bytes: bytes) -> AudioBuffer:
        """Parse a WAV byte string into an :class:`AudioBuffer`.

        Reads the actual sample rate, channel count, and sample width
        from the WAV header.  No format assumptions are made.
        """
        try:
            return AudioBuffer.from_wav(wav_bytes)
        except Exception:
            _log.exception("espeak.wav_parse_failed")
            return AudioBuffer(pcm=b"", sample_rate=self._sample_rate, channels=1)

    async def close(self) -> None:
        """No resources to release."""
        _log.info("espeak.closed")


__all__ = ["EspeakNGTTS"]
