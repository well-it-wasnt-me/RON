"""openWakeWord wake-word detector.

Uses the ``openwakeword`` library for proper wake-phrase detection
(instead of just energy-based triggering). Models are loaded lazily
on first use, similar to :class:`PiperTTS`.

Install with::

    uv pip install openwakeword

Configure with::

    DESKBOT_WAKEWORD__PROVIDER = openwakeword
    DESKBOT_WAKEWORD__THRESHOLD = 0.5

The default model is ``hey_mycroft`` which detects "Hey Mycroft".
Other built-in models include ``alexa``, ``hey_jarvis``, and ``ok_nova``.
Custom models can be loaded from ``.onnx`` files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, cast

from robot.logging import get_logger
from robot.speech.wakeword import WakeWordChecker, WakeWordDetected

_log = get_logger("speech.wakeword.openwakeword")

# openWakeWord operates on 16 kHz mono s16le audio, same as our mic.


class _OpenWakeWordModel(Protocol):
    """Minimal typed surface used from the optional openWakeWord package."""

    def predict(self, samples: object) -> dict[str, float]: ...


@dataclass(slots=True)
class OpenWakeWordChecker(WakeWordChecker):
    """Wake-word checker using the openWakeWord library.

    Parameters
    ----------
    phrase:
        The wake phrase label (e.g. ``"hey_mycroft"``, ``"alexa"``).
        Must match a model name available in openWakeWord.
    threshold:
        Detection threshold (0.0-1.0). Higher = fewer false positives
        but may miss more utterances. Default 0.5.
    model_path:
        Optional path to a custom ``.onnx`` model file. When set,
        the model is loaded from this file instead of using the
        built-in model for ``phrase``.
    """

    phrase: str = "hey_mycroft"
    threshold: float = 0.5
    model_path: str | None = None
    _model: _OpenWakeWordModel | None = field(default=None, init=False, repr=False)
    _unavailable: bool = field(default=False, init=False)
    _warmup_chunks: int = field(default=0, init=False)
    _cooldown_until_s: float = field(default=0.0, init=False)
    _buffer: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _frames_processed: int = field(default=0, init=False)
    _prediction_keys_logged: bool = field(default=False, init=False)
    #: One-shot guard so the phrase-not-found warning fires once, not every
    #: frame (the score is 0.0 for every frame when the phrase is wrong, so
    #: without this the mismatch would log thousands of times per minute).
    _phrase_mismatch_warned: bool = field(default=False, init=False)

    def _get_model(self) -> _OpenWakeWordModel:
        """Lazy-load the openWakeWord model on first use."""
        if self._model is not None:
            return self._model
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise ImportError(
                "openwakeword is required for OpenWakeWordChecker. "
                "Install it with: uv pip install openwakeword"
            ) from exc

        kwargs: dict[str, object] = {}
        if self.model_path:
            kwargs["wakeword_model_paths"] = [self.model_path]

        _log.info("openwakeword.loading", phrase=self.phrase, model_path=self.model_path)
        self._model = cast("_OpenWakeWordModel", Model(**kwargs if kwargs else {}))
        _log.info(
            "openwakeword.ready",
            phrase=self.phrase,
            threshold=self.threshold,
        )
        return self._model

    def check(self, pcm: bytes, timestamp: float) -> WakeWordDetected | None:
        """Check a chunk of s16le 16 kHz mono PCM for the wake phrase."""
        result: WakeWordDetected | None = None

        self._warmup_chunks += 1

        if self._warmup_chunks == 1:
            _log.debug(
                "openwakeword.audio_started",
                phrase=self.phrase,
                bytes=len(pcm),
            )

        if self._warmup_chunks > 3 and timestamp >= self._cooldown_until_s:
            self._buffer.extend(pcm)
            frame_bytes = 1280 * 2

            if len(self._buffer) >= frame_bytes:
                frame = bytes(self._buffer[:frame_bytes])
                del self._buffer[:frame_bytes]

                self._frames_processed += 1

                import numpy as np

                samples = np.frombuffer(frame, dtype=np.int16)

                if not self._unavailable:
                    try:
                        model = self._get_model()
                    except ImportError:
                        self._unavailable = True
                        _log.warning("openwakeword.unavailable")
                    else:
                        score = self._predict(model, samples)

                        if self._frames_processed == 1 or self._frames_processed % 25 == 0:
                            _log.debug(
                                "openwakeword.score",
                                phrase=self.phrase,
                                score=round(score, 4),
                                threshold=self.threshold,
                                frames=self._frames_processed,
                            )

                        if score >= self.threshold:
                            self._cooldown_until_s = timestamp + 1.5
                            _log.info(
                                "openwakeword.detected",
                                phrase=self.phrase,
                                score=round(score, 3),
                                timestamp=round(timestamp, 2),
                            )
                            result = WakeWordDetected(
                                phrase=self.phrase,
                                confidence=score,
                            )

        return result

    def _predict(self, model: _OpenWakeWordModel, samples: object) -> float:
        """Run prediction and return the score for our phrase."""
        predictions = model.predict(samples)

        if not self._prediction_keys_logged:
            self._prediction_keys_logged = True
            _log.info(
                "openwakeword.predictions",
                phrase=self.phrase,
                keys=sorted(str(key) for key in predictions),
            )

        target = _normalise_phrase(self.phrase)
        for key, value in predictions.items():
            model_phrase = _normalise_phrase(key)
            if target == model_phrase or target in model_phrase or model_phrase in target:
                return float(value) if isinstance(value, (int, float)) else 0.0

        # No prediction key matches the configured phrase. The score is
        # structurally 0.0 for every frame, so the wake word can never fire.
        # Warn once (loudly) with the available model names and the
        # remediation: this is the failure mode that silently disabled voice
        # wake when DESKBOT_WAKEWORD__PHRASE was set to a name with no
        # matching openWakeWord model.
        if not self._phrase_mismatch_warned:
            self._phrase_mismatch_warned = True
            _log.warning(
                "openwakeword.phrase_not_found",
                phrase=self.phrase,
                available=sorted(str(key) for key in predictions),
                message=(
                    "configured wake phrase does not match any loaded "
                    "openWakeWord model; score will always be 0.0 and the "
                    "wake word will never trigger. Set "
                    "DESKBOT_WAKEWORD__PHRASE to a loaded model name (e.g. "
                    "hey_mycroft, hey_jarvis, alexa) or point "
                    "DESKBOT_WAKEWORD__MODEL_PATH at a custom .onnx model."
                ),
            )
        return 0.0

    def reset(self) -> None:
        """Reset the warmup counter and cooldown."""
        self._warmup_chunks = 0
        self._cooldown_until_s = 0.0
        self._buffer.clear()
        self._frames_processed = 0


def _normalise_phrase(value: str) -> str:
    """Normalise human/model wake-word spellings for comparison."""
    return "".join(
        char
        for char in value.lower().replace("-", "_").replace(" ", "_")
        if char.isalnum() or char == "_"
    )


__all__ = ["OpenWakeWordChecker"]
