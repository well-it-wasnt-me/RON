"""Tests for the structured logging + dashboard ring buffer.

Covers the core bug fix (structlog events reaching the ring buffer despite
``PrintLoggerFactory`` bypassing the stdlib tree) and the server-side
filtering used by the ``/#/logs`` dashboard.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

import pytest

from robot.logging import (
    LogEntry,
    _RingBufferHandler,
    configure_logging,
    get_logger,
    get_ring_buffer,
    install_ring_buffer,
)


@pytest.fixture
def fresh_logging() -> Iterator[None]:
    """Reconfigure logging with a throwaway stream + fresh ring buffer."""
    configure_logging(None, stream=io.StringIO(), force=True)
    get_ring_buffer().clear()
    yield
    get_ring_buffer().clear()


def _entry(
    *,
    event: str = "evt",
    level: str = "INFO",
    logger_name: str = "robot.test",
    epoch: float = 1000.0,
    data: dict[str, Any] | None = None,
) -> LogEntry:
    return LogEntry(
        timestamp=f"t-{epoch}",
        created_epoch=epoch,
        level=level,
        logger_name=logger_name,
        event=event,
        data=data or {},
    )


# ---------------------------------------------------------------------------
# The bug fix: structlog events reach the ring buffer
# ---------------------------------------------------------------------------


def test_structlog_event_lands_in_ring_buffer(fresh_logging: None) -> None:
    """A structlog event must be captured even with PrintLoggerFactory."""
    log = get_logger("test_mod")
    log.info("my_event", key="value")

    entries = get_ring_buffer().get_entries()
    assert len(entries) == 1
    e = entries[0]
    assert e.event == "my_event"
    assert e.level == "INFO"
    assert e.logger_name == "robot.test_mod"
    assert e.data == {"key": "value"}
    assert e.created_epoch > 0


def test_structlog_logger_name_not_in_stdout_json(fresh_logging: None) -> None:
    """The bound logger_name is stripped before rendering to stdout."""
    stream = io.StringIO()
    configure_logging(None, stream=stream, force=True)
    get_ring_buffer().clear()
    get_logger("alpha").info("hello", x=1)
    rendered = stream.getvalue().strip()
    assert "logger_name" not in rendered
    assert '"event": "hello"' in rendered


def test_get_logger_prefixes_robot(fresh_logging: None) -> None:
    """Names without the robot. prefix get it added."""
    get_logger("beta").warning("w")
    e = get_ring_buffer().get_entries()[-1]
    assert e.logger_name == "robot.beta"
    assert e.level == "WARNING"


# ---------------------------------------------------------------------------
# Ring buffer capacity
# ---------------------------------------------------------------------------


def test_ring_buffer_capacity_honored() -> None:
    handler = _RingBufferHandler(capacity=3)
    for i in range(5):
        handler.add_entry(_entry(event=f"e{i}", epoch=float(i)))
    entries = handler.get_entries(limit=10)
    assert [e.event for e in entries] == ["e2", "e3", "e4"]


def test_install_ring_buffer_replaces_singleton() -> None:
    install_ring_buffer(capacity=10)
    first = get_ring_buffer()
    install_ring_buffer(capacity=20)
    second = get_ring_buffer()
    assert first is not second
    assert get_ring_buffer() is second


# ---------------------------------------------------------------------------
# get_entries filtering
# ---------------------------------------------------------------------------


def _populated_handler() -> _RingBufferHandler:
    handler = _RingBufferHandler(capacity=100)
    handler.add_entry(_entry(event="DisplayUpdated", level="DEBUG", logger_name="robot.face", epoch=1.0))
    handler.add_entry(_entry(event="StateChanged", level="INFO", logger_name="robot.behavior", epoch=2.0))
    handler.add_entry(_entry(event="LookRequested", level="INFO", logger_name="robot.behavior", epoch=3.0))
    handler.add_entry(_entry(event="RobotError", level="ERROR", logger_name="robot.core", epoch=4.0, data={"reason": "boom"}))
    return handler


def test_filter_by_level() -> None:
    handler = _populated_handler()
    entries = handler.get_entries(level="error")
    assert [e.event for e in entries] == ["RobotError"]


def test_filter_level_all_means_no_filter() -> None:
    handler = _populated_handler()
    assert len(handler.get_entries(level="ALL")) == 4
    assert len(handler.get_entries(level="")) == 4


def test_filter_by_logger_substring() -> None:
    handler = _populated_handler()
    entries = handler.get_entries(logger="behavior")
    assert {e.event for e in entries} == {"StateChanged", "LookRequested"}


def test_filter_by_event_substring() -> None:
    handler = _populated_handler()
    entries = handler.get_entries(event="state")
    assert [e.event for e in entries] == ["StateChanged"]


def test_filter_exclude_deny_list() -> None:
    handler = _populated_handler()
    entries = handler.get_entries(exclude=["DisplayUpdated", "LookRequested"])
    assert {e.event for e in entries} == {"StateChanged", "RobotError"}


def test_filter_search_matches_data_values() -> None:
    handler = _populated_handler()
    entries = handler.get_entries(search="boom")
    assert [e.event for e in entries] == ["RobotError"]


def test_filter_since_epoch() -> None:
    handler = _populated_handler()
    entries = handler.get_entries(since_epoch=3.0)
    assert {e.event for e in entries} == {"LookRequested", "RobotError"}


def test_filter_limit_returns_most_recent() -> None:
    handler = _populated_handler()
    entries = handler.get_entries(limit=2)
    assert [e.event for e in entries] == ["LookRequested", "RobotError"]


def test_combined_filters() -> None:
    handler = _populated_handler()
    entries = handler.get_entries(level="INFO", exclude=["LookRequested"])
    assert [e.event for e in entries] == ["StateChanged"]


# ---------------------------------------------------------------------------
# distinct_filters
# ---------------------------------------------------------------------------


def test_distinct_filters() -> None:
    handler = _populated_handler()
    filters = handler.distinct_filters()
    assert filters["levels"] == ["DEBUG", "ERROR", "INFO"]
    assert filters["loggers"] == ["robot.behavior", "robot.core", "robot.face"]
    assert filters["events"] == ["DisplayUpdated", "LookRequested", "RobotError", "StateChanged"]


def test_distinct_filters_empty() -> None:
    assert _RingBufferHandler(capacity=10).distinct_filters() == {
        "levels": [],
        "loggers": [],
        "events": [],
    }


def test_clear_empties_buffer() -> None:
    handler = _populated_handler()
    assert handler.get_entries()
    handler.clear()
    assert handler.get_entries() == []
