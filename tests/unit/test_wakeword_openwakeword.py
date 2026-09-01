"""Tests for the OpenWakeWordChecker."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from robot.speech.wakeword_openwakeword import OpenWakeWordChecker


class TestOpenWakeWordCheckerInit:
    def test_default_config(self) -> None:
        checker = OpenWakeWordChecker(phrase="hey_mycroft")
        assert checker.phrase == "hey_mycroft"
        assert checker.threshold == 0.5
        assert checker.model_path is None

    def test_custom_threshold(self) -> None:
        checker = OpenWakeWordChecker(phrase="alexa", threshold=0.8)
        assert checker.threshold == 0.8


class TestOpenWakeWordCheckerWarmup:
    def test_warmup_skips_first_chunks(self) -> None:
        """The first 3 chunks should return None (warmup period)."""
        import struct

        checker = OpenWakeWordChecker(phrase="hey_mycroft")
        pcm = struct.pack("<160h", *([1000] * 160))
        assert checker.check(pcm, 0.0) is None
        assert checker.check(pcm, 0.03) is None
        assert checker.check(pcm, 0.06) is None


class TestOpenWakeWordCheckerCooldown:
    def test_cooldown_prevents_rapid_retrigger(self) -> None:
        """After a detection, cooldown prevents immediate re-trigger."""
        import struct

        checker = OpenWakeWordChecker(phrase="hey_mycroft")
        checker._warmup_chunks = 10
        checker._cooldown_until_s = 5.0
        pcm = struct.pack("<160h", *([1000] * 160))
        assert checker.check(pcm, 4.9) is None

    def test_reset_clears_state(self) -> None:
        checker = OpenWakeWordChecker(phrase="hey_mycroft")
        checker._warmup_chunks = 100
        checker._cooldown_until_s = 50.0
        checker.reset()
        assert checker._warmup_chunks == 0
        assert checker._cooldown_until_s == 0.0


class TestOpenWakeWordCheckerMissingLibrary:
    def test_missing_library_raises_on_model_load(self) -> None:
        """If openwakeword is not installed, _get_model should raise ImportError."""
        checker = OpenWakeWordChecker(phrase="hey_mycroft")
        with (
            patch.dict("sys.modules", {"openwakeword.model": None}),
            pytest.raises(ImportError, match="openwakeword"),
        ):
            checker._get_model()


class _FakeModel:
    """Stand-in for openwakeword's Model returning a fixed prediction dict."""

    def __init__(self, predictions: dict[str, float]) -> None:
        self._predictions = predictions

    def predict(self, samples: object) -> dict[str, float]:
        return dict(self._predictions)


class TestOpenWakeWordCheckerPhraseMismatch:
    """A configured phrase with no matching model key can never fire.

    Regression for the silent failure that disabled voice wake on the Pi:
    ``DESKBOT_WAKEWORD__PHRASE=hey ron`` loaded only the built-in models
    (alexa, hey_mycroft, ...), so the score was 0.0 for every frame and
    the wake word never triggered -- with no warning. The checker must
    now flag this once, loudly.
    """

    def test_mismatched_phrase_warns_once_and_scores_zero(self) -> None:
        import struct

        # Configured phrase "hey ron" but the model only scores "hey_mycroft".
        checker = OpenWakeWordChecker(phrase="hey ron", threshold=0.5)
        checker._model = _FakeModel({"hey_mycroft": 0.9})
        checker._warmup_chunks = 10  # skip warmup

        frame = struct.pack(f"<{1280}h", *([0] * 1280))

        # First mismatched frame: no detection, and the one-shot warning arms.
        assert checker.check(frame, 0.08) is None
        assert checker._phrase_mismatch_warned is True

        # Subsequent frames must not re-warn (the guard is one-shot).
        assert checker.check(frame, 0.16) is None
        assert checker.check(frame, 0.24) is None
        # Still armed, never reset -- the flag is monotonic.
        assert checker._phrase_mismatch_warned is True

    def test_matching_phrase_does_not_warn(self) -> None:
        import struct

        checker = OpenWakeWordChecker(phrase="hey_mycroft", threshold=0.5)
        checker._model = _FakeModel({"hey_mycroft": 0.9})
        checker._warmup_chunks = 10

        frame = struct.pack(f"<{1280}h", *([0] * 1280))
        result = checker.check(frame, 0.08)
        assert result is not None
        assert result.phrase == "hey_mycroft"
        assert checker._phrase_mismatch_warned is False
