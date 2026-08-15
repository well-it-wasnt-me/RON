"""Tests for :mod:`robot.config`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from robot.config import (
    AppSettings,
    LLMConfig,
    PersonalityConfig,
    load_settings,
)


def test_personality_defaults() -> None:
    p = PersonalityConfig()
    assert 0.0 <= p.curiosity <= 1.0
    assert 0.0 <= p.energy <= 1.0


def test_personality_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        PersonalityConfig(curiosity=1.5)


def test_settings_load_with_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKBOT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DESKBOT_LLM__MODEL", "gpt-4o-mini")
    settings = load_settings()
    assert settings.log_level == "DEBUG"
    assert settings.llm.model == "gpt-4o-mini"


def test_settings_use_mocks_default() -> None:
    # Use _env_file=None to test the true defaults, not the local .env.
    settings = AppSettings(_env_file=None, timezone="UTC")
    assert settings.use_mocks is True
    assert settings.hardware == "mock"


def test_llm_temperature_bounds() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(temperature=3.0)


def test_app_settings_timezone_required(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use _env_file=None so the .env file doesn't override our init value.
    from robot.errors import ConfigurationError

    monkeypatch.delenv("DESKBOT_TIMEZONE", raising=False)
    with pytest.raises(ConfigurationError):
        AppSettings(_env_file=None, timezone="")


def test_app_settings_env_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DESKBOT_ENV", raising=False)
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, env="staging")
