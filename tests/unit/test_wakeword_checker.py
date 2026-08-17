"""Tests for the wake-word checker protocol and implementations."""

from __future__ import annotations

import os
import struct
from typing import Any

import pytest

from robot.events.events import WakeWordDetected
from robot.speech.wakeword import (
    AudioActivityDetector,
    MockWakeWordChecker,
    NullWakeWordChecker,
    WakeWordChecker,
)
from robot.speech.wakeword_energy import EnergyActivityDetector

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_null_checker_is_wake_word_checker() -> None:
    assert isinstance(NullWakeWordChecker(), WakeWordChecker)


def test_mock_checker_is_wake_word_checker() -> None:
    assert isinstance(MockWakeWordChecker(), WakeWordChecker)


def test_energy_activity_detector_is_not_a_wake_word_checker() -> None:
    """Energy/RMS detection is VAD, not semantic wake-word detection."""
    detector = EnergyActivityDetector(threshold=0.01, warmup_chunks=0)
    assert isinstance(detector, AudioActivityDetector)
    assert not isinstance(detector, WakeWordChecker)


# ---------------------------------------------------------------------------
# NullWakeWordChecker
# ---------------------------------------------------------------------------


def test_null_checker_always_returns_none() -> None:
    checker = NullWakeWordChecker()
    assert checker.check(b"\x00" * 100, 1.0) is None
    assert checker.check(b"\xff" * 100, 2.0) is None


# ---------------------------------------------------------------------------
# MockWakeWordChecker
# ---------------------------------------------------------------------------


def test_mock_checker_triggers_after_n_chunks() -> None:
    checker = MockWakeWordChecker(phrase="test", trigger_after_chunks=3)

    assert checker.check(b"\x00" * 100, 0.0) is None
    assert checker.check(b"\x00" * 100, 0.1) is None

    result = checker.check(b"\x00" * 100, 0.2)

    assert result is not None
    assert isinstance(result, WakeWordDetected)
    assert result.phrase == "test"
    assert result.confidence == 1.0


def test_mock_checker_resets_after_trigger() -> None:
    checker = MockWakeWordChecker(trigger_after_chunks=2)

    assert checker.check(b"\x00" * 100, 0.0) is None

    result1 = checker.check(b"\x00" * 100, 0.1)
    assert result1 is not None

    assert checker.check(b"\x00" * 100, 0.2) is None

    result2 = checker.check(b"\x00" * 100, 0.3)
    assert result2 is not None


# ---------------------------------------------------------------------------
# EnergyActivityDetector
# ---------------------------------------------------------------------------


def test_energy_activity_skips_warmup() -> None:
    """The configured warmup chunks are always ignored."""
    detector = EnergyActivityDetector(threshold=0.01, warmup_chunks=3)
    loud = struct.pack("<100h", *([16000] * 100))

    for i in range(3):
        assert detector.is_active(loud, float(i) * 0.01) is False


def test_energy_activity_detects_loud_signal() -> None:
    detector = EnergyActivityDetector(threshold=0.05, warmup_chunks=0)
    loud = struct.pack("<100h", *([16000] * 100))

    assert detector.is_active(loud, 0.1) is True


def test_energy_activity_ignores_quiet_signal() -> None:
    detector = EnergyActivityDetector(threshold=0.5, warmup_chunks=0)
    quiet = b"\x00" * 200

    assert detector.is_active(quiet, 0.1) is False


def test_energy_activity_cooldown() -> None:
    """A second active report inside cooldown_s is suppressed."""
    detector = EnergyActivityDetector(
        threshold=0.05,
        cooldown_s=2.0,
        warmup_chunks=0,
    )
    loud = struct.pack("<100h", *([16000] * 100))

    assert detector.is_active(loud, 0.1) is True
    assert detector.is_active(loud, 0.5) is False
    assert detector.is_active(loud, 2.5) is True


def test_energy_activity_reset() -> None:
    """reset() clears warmup and cooldown state."""
    detector = EnergyActivityDetector(
        threshold=0.05,
        warmup_chunks=5,
        cooldown_s=10.0,
    )
    loud = struct.pack("<100h", *([16000] * 100))

    for i in range(5):
        detector.is_active(b"\x00" * 100, float(i) * 0.01)

    assert detector.is_active(loud, 1.0) is True

    detector.reset()

    assert detector.is_active(loud, 0.1) is False


def test_energy_activity_first_report_not_blocked_by_cooldown() -> None:
    """The first active report is not blocked by initial cooldown state."""
    detector = EnergyActivityDetector(
        threshold=0.05,
        cooldown_s=10.0,
        warmup_chunks=0,
    )
    loud = struct.pack("<100h", *([16000] * 100))

    assert detector.is_active(loud, 0.5) is True


# ---------------------------------------------------------------------------
# ConversationService integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversation_service_with_mock_checker() -> None:
    """ConversationService uses the injected WakeWordChecker."""
    from robot.ai.conversation import ConversationManager
    from robot.ai.llm_mock import MockLLM
    from robot.ai.prompts import system_prompt
    from robot.behavior.state_machine import RobotState, StateMachine
    from robot.events.bus import InMemoryEventBus
    from robot.services.conversation_service import ConversationService
    from robot.speech.stt import MockSTT
    from robot.speech.tts import MockTTS

    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    sm._state = RobotState.IDLE

    llm = MockLLM()
    llm.register("hello", "Hi there!")

    stt = MockSTT(transcript="hello")
    tts = MockTTS()
    conversation = ConversationManager(
        llm=llm,
        system_prompt=system_prompt(),
    )
    checker = MockWakeWordChecker(trigger_after_chunks=1)

    svc = ConversationService(
        bus=bus,
        state_machine=sm,
        stt=stt,
        tts=tts,
        llm=llm,
        conversation=conversation,
        wake_checker=checker,
    )

    assert svc.wake_checker is checker


@pytest.mark.asyncio
async def test_conversation_service_null_checker_by_default() -> None:
    """A missing wake checker defaults to NullWakeWordChecker."""
    from robot.ai.conversation import ConversationManager
    from robot.ai.llm_mock import MockLLM
    from robot.ai.prompts import system_prompt
    from robot.behavior.state_machine import StateMachine
    from robot.events.bus import InMemoryEventBus
    from robot.services.conversation_service import ConversationService
    from robot.speech.stt import MockSTT
    from robot.speech.tts import MockTTS

    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    llm = MockLLM()
    stt = MockSTT()
    tts = MockTTS()
    conversation = ConversationManager(
        llm=llm,
        system_prompt=system_prompt(),
    )

    svc = ConversationService(
        bus=bus,
        state_machine=sm,
        stt=stt,
        tts=tts,
        llm=llm,
        conversation=conversation,
    )

    assert isinstance(svc.wake_checker, NullWakeWordChecker)


# ---------------------------------------------------------------------------
# WakeWordConfig
# ---------------------------------------------------------------------------


class TestWakeWordConfigPorcupineSnowboy:
    def test_porcupine_defaults(self) -> None:
        from robot.config import WakeWordConfig

        cfg = WakeWordConfig()

        assert cfg.porcupine_access_key == ""
        assert cfg.porcupine_keyword == "picovoice"
        assert cfg.porcupine_model_path == ""
        assert cfg.snowboy_model_path == ""

    def test_porcupine_env_override(self) -> None:
        from robot.config import WakeWordConfig

        env: dict[str, str] = {
            "DESKBOT_WAKEWORD__PORCUPINE_ACCESS_KEY": "test-key-123",
            "DESKBOT_WAKEWORD__PORCUPINE_KEYWORD": "hey google",
            "DESKBOT_WAKEWORD__PORCUPINE_MODEL_PATH": "/path/to/model.ppn",
            "DESKBOT_WAKEWORD__SNOWBOY_MODEL_PATH": "/path/to/model.pmdl",
        }

        original: dict[str, str | None] = {key: os.environ.get(key) for key in env}

        for key, value in env.items():
            os.environ[key] = value

        try:
            cfg = WakeWordConfig()

            assert cfg.porcupine_access_key == "test-key-123"
            assert cfg.porcupine_keyword == "hey google"
            assert cfg.porcupine_model_path == "/path/to/model.ppn"
            assert cfg.snowboy_model_path == "/path/to/model.pmdl"
        finally:
            for key, value in original.items():  # type: ignore[assignment]
                if value is None:
                    os.environ.pop(key, None)  # type: ignore[unreachable]
                else:
                    os.environ[key] = value

    def test_provider_accepts_porcupine(self) -> None:
        from robot.config import WakeWordConfig

        cfg = WakeWordConfig(provider="porcupine")
        assert cfg.provider == "porcupine"

    def test_provider_accepts_snowboy(self) -> None:
        from robot.config import WakeWordConfig

        cfg = WakeWordConfig(provider="snowboy")
        assert cfg.provider == "snowboy"

    def test_energy_provider_is_rejected(self) -> None:
        """The obsolete energy provider must be rejected."""
        from pydantic import ValidationError

        from robot.config import WakeWordConfig

        with pytest.raises(ValidationError):
            WakeWordConfig(provider="energy")

    def test_energy_fields_are_removed(self) -> None:
        """Obsolete energy configuration fields must not exist."""
        from robot.config import WakeWordConfig

        cfg = WakeWordConfig()

        assert not hasattr(cfg, "energy_threshold")
        assert not hasattr(cfg, "energy_cooldown_s")
        assert not hasattr(cfg, "energy_warmup_chunks")


# ---------------------------------------------------------------------------
# Wake-word factory
# ---------------------------------------------------------------------------


class TestWakeWordFactoryProviders:
    """Test the app factory's wake-word backend selection."""

    def test_mock_provider(self) -> None:
        from robot.app import _build_ai_stack
        from robot.behavior.state_machine import StateMachine
        from robot.config import AppSettings
        from robot.events.bus import InMemoryEventBus

        settings = AppSettings(
            _env_file=None,
            env="testing",
            log_level="WARNING",
        )
        settings.wakeword.provider = "mock"

        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)

        cs = _build_ai_stack(
            bus=bus,
            state_machine=sm,
            microphone=None,
            settings=settings,
        )

        assert isinstance(cs.wake_checker, MockWakeWordChecker)

    def test_openwakeword_provider_creates_real_checker(self) -> None:
        """openwakeword creates the real checker when installed."""
        from robot.app import _build_ai_stack
        from robot.behavior.state_machine import StateMachine
        from robot.config import AppSettings
        from robot.events.bus import InMemoryEventBus
        from robot.speech.wakeword_openwakeword import OpenWakeWordChecker

        settings = AppSettings(
            _env_file=None,
            env="testing",
            log_level="WARNING",
        )
        settings.wakeword.provider = "openwakeword"

        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)

        cs = _build_ai_stack(
            bus=bus,
            state_machine=sm,
            microphone=None,
            settings=settings,
        )

        assert isinstance(cs.wake_checker, OpenWakeWordChecker)

    def test_porcupine_provider_falls_back_without_package(self) -> None:
        """Porcupine falls back to NullWakeWordChecker if unavailable."""
        from robot.app import _build_ai_stack
        from robot.behavior.state_machine import StateMachine
        from robot.config import AppSettings
        from robot.events.bus import InMemoryEventBus

        settings = AppSettings(
            _env_file=None,
            env="testing",
            log_level="WARNING",
        )
        settings.wakeword.provider = "porcupine"
        settings.wakeword.porcupine_access_key = "test-key"

        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)

        cs = _build_ai_stack(
            bus=bus,
            state_machine=sm,
            microphone=None,
            settings=settings,
        )

        assert isinstance(cs.wake_checker, NullWakeWordChecker)

    def test_snowboy_provider_falls_back_without_package(self) -> None:
        """Snowboy falls back to NullWakeWordChecker if unavailable."""
        from robot.app import _build_ai_stack
        from robot.behavior.state_machine import StateMachine
        from robot.config import AppSettings
        from robot.events.bus import InMemoryEventBus

        settings = AppSettings(
            _env_file=None,
            env="testing",
            log_level="WARNING",
        )
        settings.wakeword.provider = "snowboy"
        settings.wakeword.snowboy_model_path = "/path/to/model.pmdl"

        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)

        cs = _build_ai_stack(
            bus=bus,
            state_machine=sm,
            microphone=None,
            settings=settings,
        )

        assert isinstance(cs.wake_checker, NullWakeWordChecker)

    def test_energy_provider_never_produces_energy_checker(self) -> None:
        """An obsolete provider must never create an energy checker."""
        from robot.app import _build_ai_stack
        from robot.behavior.state_machine import StateMachine
        from robot.config import AppSettings
        from robot.events.bus import InMemoryEventBus

        settings = AppSettings(
            _env_file=None,
            env="testing",
            log_level="WARNING",
        )

        # Pydantic validates the public assignment. Bypass validation only
        # for this test so that the factory's runtime fallback is exercised.
        wakeword: Any = settings.wakeword
        wakeword.provider = "energy"

        bus = InMemoryEventBus()
        sm = StateMachine(bus=bus)

        cs = _build_ai_stack(
            bus=bus,
            state_machine=sm,
            microphone=None,
            settings=settings,
        )

        assert isinstance(cs.wake_checker, NullWakeWordChecker)
