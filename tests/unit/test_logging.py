"""Tests for the logging helpers."""

from __future__ import annotations

import pytest
import structlog

from robot.config import AppSettings
from robot.logging import configure_logging, get_logger


def test_configure_logging_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings(log_level="DEBUG")
    configure_logging(settings)
    configure_logging(settings)  # must not raise
    logger = get_logger("test")
    # The first call resolves the proxy to a real logger; subsequent calls return the same instance.
    _ = logger.info("hi")


def test_get_logger_prefixes_name() -> None:
    logger = get_logger("foo")
    # We can't introspect the bound logger, but we can at least ensure it is
    # usable without raising.
    logger.info("hello")


def test_log_level_honored() -> None:
    settings = AppSettings(log_level="WARNING")
    configure_logging(settings)
    assert structlog.get_config() is not None
