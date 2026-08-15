"""Tests for the DisplayConfig Pydantic model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from robot.config import DisplayConfig


def test_defaults_match_wiring_md() -> None:
    """The defaults should match docs/wiring.md (BCM 25 D/C, BCM 24 RST)."""
    c = DisplayConfig(backend="gc9a01")
    assert c.dc_pin == 25
    assert c.reset_pin == 24
    assert c.backlight_pin is None
    assert c.spi_hz == 8_000_000
    assert c.spi_mode == 0
    assert c.chunk_bytes == 4096
    assert c.invert is True
    assert c.rotation == 0


def test_invalid_spi_mode_rejected() -> None:
    from robot.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        DisplayConfig(backend="gc9a01", spi_mode=4)


def test_spi_hz_below_minimum_rejected() -> None:
    with pytest.raises(ValidationError):
        DisplayConfig(backend="gc9a01", spi_hz=1_000)


def test_chunk_bytes_below_minimum_rejected() -> None:
    with pytest.raises(ValidationError):
        DisplayConfig(backend="gc9a01", chunk_bytes=8)


def test_validate_pins_allows_unique() -> None:
    c = DisplayConfig(backend="gc9a01", dc_pin=25, reset_pin=24, backlight_pin=18)
    c.validate_pins()  # returns None, but we only care it does not raise


def test_validate_pins_rejects_collision() -> None:
    from robot.errors import ConfigurationError

    c = DisplayConfig(backend="gc9a01", dc_pin=25, reset_pin=25)
    with pytest.raises(ConfigurationError):
        c.validate_pins()


def test_effective_spi_hz_returns_configured_value() -> None:
    c = DisplayConfig(backend="gc9a01", spi_hz=16_000_000)
    assert c.effective_spi_hz() == 16_000_000


def test_env_overrides_apply() -> None:
    """Environment variables override defaults (tested without .env)."""
    import os

    os.environ["DESKBOT_DISPLAYS__SPI_HZ"] = "16000000"
    os.environ["DESKBOT_DISPLAYS__SPI_MODE"] = "3"
    os.environ["DESKBOT_DISPLAYS__INVERT"] = "false"
    os.environ["DESKBOT_DISPLAYS__RESET_PIN"] = "23"
    try:
        c = DisplayConfig(backend="gc9a01")
        assert c.spi_hz == 16_000_000
        assert c.spi_mode == 3
        assert c.invert is False
        assert c.reset_pin == 23
    finally:
        for key in (
            "DESKBOT_DISPLAYS__SPI_HZ",
            "DESKBOT_DISPLAYS__SPI_MODE",
            "DESKBOT_DISPLAYS__INVERT",
            "DESKBOT_DISPLAYS__RESET_PIN",
        ):
            os.environ.pop(key, None)
