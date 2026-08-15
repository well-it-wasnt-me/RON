"""Snowboy wake-word detector.

Uses the Snowboy hotword detection engine for offline, low-latency
wake word detection. Snowboy is a lightweight engine suitable for
Raspberry Pi deployment.

.. note::
   Snowboy is no longer actively maintained. Consider Porcupine or
   openWakeWord for new projects. This module is provided for
   compatibility with existing Snowboy models.
"""

from __future__ import annotations

from robot.logging import get_logger
from robot.speech.wakeword import WakeWordChecker, WakeWordDetected

_log = get_logger("speech.wakeword_snowboy")


class SnowboyWakeWordChecker(WakeWordChecker):
    """Wake-word detection using Snowboy.

    Requires the ``snowboy-snowboy`` or ``snowboy`` package. Snowboy
    uses pre-trained model files (``.pmdl`` or ``.umdl``) for wake
    word detection.
    """

    def __init__(
        self,
        model_path: str,
        sensitivity: float = 0.5,
        audio_gain: float = 1.0,
    ) -> None:
        try:
            import snowboydecoder  # type: ignore[import-not-found]
            import snowboydetect  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "snowboy is required for SnowboyWakeWordChecker. Install with: pip install snowboy"
            ) from exc

        self._model_path = model_path
        self._sensitivity = sensitivity
        self._audio_gain = audio_gain

        # snowboydetect.SnowboyDetect is the low-level detector.
        self._detector = snowboydetect.SnowboyDetect(
            resource_filename=snowboydecoder.RESOURCE_FILE.encode(),
            model_str=model_path.encode(),
        )
        self._detector.SetSensitivity(str(sensitivity).encode())
        self._detector.SetAudioGain(audio_gain)

        self._sample_rate = self._detector.SampleRate()
        self._frame_length = self._detector.NumChannels()
        _log.info(
            "wakeword.snowboy.initialized",
            model=model_path,
            sensitivity=sensitivity,
            sample_rate=self._sample_rate,
        )

    @property
    def sample_rate(self) -> int:
        return int(self._sample_rate)

    @property
    def frame_length(self) -> int:
        return int(self._frame_length)

    def check(self, pcm: bytes, timestamp: float) -> WakeWordDetected | None:
        """Check a PCM audio frame for the wake word.

        Parameters
        ----------
        pcm:
            Raw s16le mono PCM bytes.
        timestamp:
            Monotonic timestamp in seconds.
        """
        result = self._detector.RunDetection(pcm)
        if result > 0:
            _log.info("wakeword.snowboy.detected", model=self._model_path)
            return WakeWordDetected(phrase=self._model_path, confidence=self._sensitivity)
        return None

    def close(self) -> None:
        """Release Snowboy resources."""
        if hasattr(self, "_detector") and self._detector is not None:
            del self._detector
            _log.info("wakeword.snowboy.closed")
