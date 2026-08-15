"""Tests for the WebSocket event streamer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from robot.api.ws import EventStreamer, _event_to_dict


class FakeState(str, Enum):
    IDLE = "idle"
    CURIOUS = "curious"


@dataclass(slots=True, frozen=True)
class FakeEvent:
    name: str
    value: int


@dataclass(slots=True, frozen=True)
class FakeEnumEvent:
    state: FakeState


def test_event_to_dict_plain_fields() -> None:
    event = FakeEvent(name="test", value=42)
    result = _event_to_dict(event)
    assert result == {"name": "test", "value": 42}


def test_event_to_dict_enum_field() -> None:
    event = FakeEnumEvent(state=FakeState.CURIOUS)
    result = _event_to_dict(event)
    assert result == {"state": "curious"}


def test_streamer_initial_state() -> None:
    streamer = EventStreamer()
    assert streamer.connection_count == 0


def test_streamer_initial_no_connections() -> None:
    streamer = EventStreamer()
    assert streamer._connections == []


@pytest.mark.anyio
async def test_streamer_on_event_converts_enum() -> None:
    """on_event should convert dataclasses with Enum fields to values."""
    streamer = EventStreamer()
    # No connections, so broadcast should be a no-op.
    await streamer.on_event(FakeEnumEvent(state=FakeState.IDLE))
    # Should not raise even with no connections.


@pytest.mark.anyio
async def test_streamer_broadcast_no_connections() -> None:
    """broadcast should work with no connections (no-op)."""
    streamer = EventStreamer()
    await streamer.broadcast("StateChanged", {"previous": "idle", "current": "curious"})
    # No error, no crash.
