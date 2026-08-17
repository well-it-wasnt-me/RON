"""Backend/degradation tests for laptop speech I/O configuration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from robot.app import DeskBotApp
from robot.config import AppSettings, load_settings
from robot.interfaces.audio import AudioBuffer


class _RealMic:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.sample_rate = 16_000
        self.channels = 1
        self.frame_ms = 30

    async def stream(self) -> AsyncIterator[bytes]:
        for _ in ():
            yield b""

    async def close(self) -> None:
        return None


class _RealCamera:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.width = 640
        self.height = 480

    async def capture(self) -> object:
        raise RuntimeError("not used")

    async def close(self) -> None:
        return None


class _RealAudio:
    sample_rate = 48_000
    channels = 1
    output_device = "default"

    async def play(self, buffer: AudioBuffer) -> None:
        del buffer

    async def stop(self) -> None:
        return None

    async def close(self) -> None:
        return None


class _RealTTS:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    async def speak(self, text: str) -> AudioBuffer:
        del text
        return AudioBuffer(
            pcm=b"\x00\x01",
            sample_rate=24_000,
            channels=1,
        )

    async def close(self) -> None:
        return None


def _settings() -> AppSettings:
    settings = load_settings()
    settings.hardware = "real"
    settings.perception.enabled = False
    settings.learning.enabled = False
    settings.audio.backend = "usb"
    settings.tts.provider = "openai"
    settings.memory.enabled = False
    settings.vector_memory.enabled = False
    settings.conversation.store = "memory"
    settings.preferences.store = "memory"
    return settings


def _report(app: DeskBotApp) -> dict[str, Any]:
    assert app._degradation is not None
    return {entry.component: entry for entry in app._degradation.report()}


def test_real_microphone_real_tts_real_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()

    monkeypatch.setattr("robot.app.UsbMicrophone", _RealMic)
    monkeypatch.setattr("robot.app.UsbCamera", _RealCamera)
    monkeypatch.setattr("robot.app.OpenAITTS", _RealTTS)
    monkeypatch.setattr(
        "robot.app._usb_speaker",
        lambda _settings: _RealAudio(),
    )

    app = DeskBotApp.build(settings)

    assert app._degradation is not None
    assert app.conversation is not None

    report = _report(app)

    assert type(app._audio).__name__ == "_RealAudio"
    assert type(app.conversation.tts).__name__ == "_RealTTS"
    assert report["audio"].status == "ok"
    assert report["tts"].status == "ok"
    assert report["microphone"].status == "ok"

    app._close_stores()


def test_real_microphone_failed_tts_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()

    monkeypatch.setattr("robot.app.UsbMicrophone", _RealMic)
    monkeypatch.setattr("robot.app.UsbCamera", _RealCamera)
    monkeypatch.setattr(
        "robot.app._usb_speaker",
        lambda _settings: _RealAudio(),
    )

    def _raise_tts(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("tts init failed")

    monkeypatch.setattr("robot.app.OpenAITTS", _raise_tts)

    app = DeskBotApp.build(settings)

    assert app._degradation is not None
    assert app.conversation is not None

    report = _report(app)

    assert type(app.conversation.tts).__name__ == "MockTTS"
    assert report["tts"].status == "degraded"
    assert report["audio"].status == "ok"

    app._close_stores()


def test_real_microphone_failed_audio_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()

    monkeypatch.setattr("robot.app.UsbMicrophone", _RealMic)
    monkeypatch.setattr("robot.app.UsbCamera", _RealCamera)
    monkeypatch.setattr("robot.app.OpenAITTS", _RealTTS)

    def _raise_audio(_settings: AppSettings) -> None:
        raise RuntimeError("audio init failed")

    monkeypatch.setattr("robot.app._usb_speaker", _raise_audio)

    app = DeskBotApp.build(settings)

    assert app._degradation is not None
    assert app.conversation is not None

    report = _report(app)

    assert type(app._audio).__name__ == "MockAudioOutput"
    assert report["audio"].status == "degraded"
    assert report["tts"].status == "ok"

    app._close_stores()


def test_mock_microphone_mock_tts_mock_audio() -> None:
    settings = load_settings()

    settings.hardware = "mock"
    settings.audio.backend = "mock"
    settings.tts.provider = "mock"
    settings.perception.enabled = False
    settings.learning.enabled = False
    settings.memory.enabled = False
    settings.vector_memory.enabled = False
    settings.conversation.store = "memory"
    settings.preferences.store = "memory"

    app = DeskBotApp.build(settings)

    assert app._degradation is not None
    assert app.conversation is not None

    report = _report(app)

    assert type(app._audio).__name__ == "MockAudioOutput"
    assert type(app.conversation.tts).__name__ == "MockTTS"
    assert report["audio"].status == "ok"
    assert report["tts"].status == "ok"

    app._close_stores()
