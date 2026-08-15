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
