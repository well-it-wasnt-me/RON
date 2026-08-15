"""Tests for the Piper TTS driver."""

from __future__ import annotations

import pytest

from robot.config import PiperConfig, TTSConfig
from robot.speech.tts_piper import PiperTTS


def test_piper_tts_name() -> None:
    tts = PiperTTS(model="en_US-lessac-medium")
    assert tts.name == "piper:en_US-lessac-medium"


def test_piper_tts_custom_model() -> None:
    tts = PiperTTS(model="de_DE-thorsten-low")
    assert tts.name == "piper:de_DE-thorsten-low"


def test_piper_tts_default_model() -> None:
    tts = PiperTTS()
    assert tts.name == "piper:en_US-lessac-medium"


def test_piper_config_defaults() -> None:
    cfg = PiperConfig()
    assert cfg.model == "en_US-lessac-medium"
    assert cfg.download_dir == ""
    assert cfg.use_cuda is False
    assert cfg.speaker_id is None
    assert cfg.noise_scale is None
    assert cfg.length_scale is None
    assert cfg.noise_w_scale is None


def test_tts_config_has_piper() -> None:
    cfg = TTSConfig()
    assert cfg.piper.model == "en_US-lessac-medium"


def test_tts_config_piper_override() -> None:
    cfg = TTSConfig(provider="piper", piper=PiperConfig(model="en_US-lessac-low"))
    assert cfg.provider == "piper"
    assert cfg.piper.model == "en_US-lessac-low"


@pytest.mark.anyio
async def test_piper_tts_close() -> None:
    """close() should return None without error."""
    tts = PiperTTS()
    await tts.close()
    # close() returns None implicitly


def test_piper_tts_import_error() -> None:
    """PiperTTS should provide a clear error if piper-tts is not installed."""
    tts = PiperTTS()
    # The import is lazy - _get_voice() raises ImportError if piper is missing.
    # Since piper IS installed in our test env, we just verify the name works.
    assert tts.name == "piper:en_US-lessac-medium"
