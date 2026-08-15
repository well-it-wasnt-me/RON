"""Porcupine wake-word detector.

Uses Picovoice's Porcupine engine for accurate, low-latency wake word
detection. Porcupine is a commercial product with a free tier for
hobbyist use. This module requires the ``pvporcupine`` package.
"""

from __future__ import annotations

from robot.logging import get_logger
from robot.speech.wakeword import WakeWordChecker, WakeWordDetected

_log = get_logger("speech.wakeword_porcupine")


class PorcupineWakeWordChecker(WakeWordChecker):
    """Wake-word detection using Picovoice Porcupine.

    Requires the ``pvporcupine`` package::

        pip install pvporcupine

    You will need a Picovoice AccessKey (free tier available at
    https://console.picovoice.ai/) and may use built-in keywords or
    custom wake word models.
    """

    def __init__(
        self,
        access_key: str,
        keyword: str = "picovoice",
        model_path: str | None = None,
        sensitivity: float = 0.5,
    ) -> None:
        try:
            import pvporcupine  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "pvporcupine is required for PorcupineWakeWordChecker. "
                "Install with: pip install pvporcupine"
            ) from exc

        self._keyword = keyword
        self._sensitivity = sensitivity

        if model_path:
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[model_path],
                sensitivities=[sensitivity],
            )
        else:
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[keyword],
                sensitivities=[sensitivity],
            )

        self._frame_length = self._porcupine.frame_length
        self._sample_rate = self._porcupine.sample_rate
        _log.info(
            "wakeword.porcupine.initialized",
            keyword=keyword,
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
            Raw s16le mono PCM bytes. Must be exactly ``frame_length``
            samples (``frame_length * 2`` bytes).
        timestamp:
            Monotonic timestamp in seconds.
        """
        import struct

        # Convert bytes to int16 samples.
        num_samples = len(pcm) // 2
        if num_samples != self._frame_length:
            _log.warning(
                "wakeword.porcupine.frame_size_mismatch",
                expected=self._frame_length,
                got=num_samples,
            )
            return None

        samples = list(struct.unpack(f"<{num_samples}h", pcm))

        keyword_index = self._porcupine.process(samples)
        if keyword_index >= 0:
            _log.info("wakeword.porcupine.detected", keyword=self._keyword)
            return WakeWordDetected(phrase=self._keyword, confidence=self._sensitivity)

        return None

    def close(self) -> None:
        """Release Porcupine resources."""
        if hasattr(self, "_porcupine") and self._porcupine is not None:
            self._porcupine.delete()
            _log.info("wakeword.porcupine.closed")
