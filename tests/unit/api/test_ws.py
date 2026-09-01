"""Tests for the WebSocket event streamer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

import pytest
from starlette.websockets import WebSocket

from robot.api.ws import (
    EventFilter,
    EventStreamer,
    _event_to_dict,
    _parse_filter_message,
    _split_csv,
)


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


# ---------------------------------------------------------------------------
# EventFilter
# ---------------------------------------------------------------------------


def test_filter_default_matches_everything() -> None:
    f = EventFilter()
    assert f.matches("StateChanged")
    assert f.matches("DisplayUpdated")


def test_filter_include_only() -> None:
    f = EventFilter(include={"StateChanged", "EmotionChanged"})
    assert f.matches("StateChanged")
    assert not f.matches("DisplayUpdated")


def test_filter_exclude_only() -> None:
    f = EventFilter(exclude={"DisplayUpdated"})
    assert not f.matches("DisplayUpdated")
    assert f.matches("StateChanged")


def test_filter_include_wins_over_exclude() -> None:
    """When both are set, include takes precedence and exclude is ignored."""
    f = EventFilter(include={"StateChanged"}, exclude={"StateChanged"})
    assert f.matches("StateChanged") is True


def test_split_csv() -> None:
    assert _split_csv(None) is None
    assert _split_csv("") is None
    assert _split_csv("StateChanged,EmotionChanged") == {"StateChanged", "EmotionChanged"}
    assert _split_csv(" a , ,b ") == {"a", "b"}


def test_parse_filter_message_runtime_update() -> None:
    f = _parse_filter_message('{"filter": {"include": ["StateChanged"], "exclude": ["DisplayUpdated"]}}')
    assert f is not None
    assert f.include == {"StateChanged"}
    assert f.exclude == {"DisplayUpdated"}


def test_parse_filter_message_ignores_non_filter() -> None:
    assert _parse_filter_message("ping") is None
    assert _parse_filter_message('{"type": "StateChanged"}') is None
    assert _parse_filter_message("not-json") is None


def test_parse_filter_message_partial() -> None:
    f = _parse_filter_message('{"filter": {"exclude": ["DisplayUpdated"]}}')
    assert f is not None
    assert f.include is None
    assert f.exclude == {"DisplayUpdated"}


# ---------------------------------------------------------------------------
# Per-connection filtering with a fake WebSocket
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """Minimal WebSocket double that records sent text and never blocks."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def receive_text(self) -> str:
        # Tests don't drive the receive loop; park forever.
        import anyio
        await anyio.sleep(60)
        return ""


def _ws(fake: FakeWebSocket) -> WebSocket:
    """Present a FakeWebSocket as a WebSocket to the streamer's typed API."""
    return cast("WebSocket", fake)


@pytest.mark.anyio
async def test_streamer_filters_per_connection() -> None:
    """An excluded event is not sent; a non-excluded event is."""
    streamer = EventStreamer()
    noisy = FakeWebSocket()
    all_events = FakeWebSocket()
    await streamer.connect(_ws(noisy), EventFilter(exclude={"DisplayUpdated"}))
    await streamer.connect(_ws(all_events), EventFilter())

    await streamer.broadcast("DisplayUpdated", {"frame": 1})
    await streamer.broadcast("StateChanged", {"current": "curious"})

    # Noisy client only got the state change.
    assert len(noisy.sent) == 1
    assert '"StateChanged"' in noisy.sent[0]
    # Unfiltered client got both.
    assert len(all_events.sent) == 2


@pytest.mark.anyio
async def test_streamer_include_filter() -> None:
    streamer = EventStreamer()
    ws = FakeWebSocket()
    await streamer.connect(_ws(ws), EventFilter(include={"StateChanged"}))

    await streamer.broadcast("DisplayUpdated", {})
    await streamer.broadcast("StateChanged", {"current": "idle"})

    assert len(ws.sent) == 1
    assert '"StateChanged"' in ws.sent[0]


@pytest.mark.anyio
async def test_streamer_set_filter_updates_runtime() -> None:
    streamer = EventStreamer()
    ws = FakeWebSocket()
    await streamer.connect(_ws(ws), EventFilter())

    # Initially receives everything.
    await streamer.broadcast("DisplayUpdated", {})
    assert len(ws.sent) == 1

    # Update the filter to exclude DisplayUpdated.
    streamer.set_filter(_ws(ws), EventFilter(exclude={"DisplayUpdated"}))
    await streamer.broadcast("DisplayUpdated", {})
    assert len(ws.sent) == 1  # unchanged: filtered out


@pytest.mark.anyio
async def test_streamer_disconnect_removes_connection() -> None:
    streamer = EventStreamer()
    ws = FakeWebSocket()
    await streamer.connect(_ws(ws), EventFilter())
    assert streamer.connection_count == 1
    streamer.disconnect(_ws(ws))
    assert streamer.connection_count == 0
    # Broadcast after disconnect is a no-op (no stale sends).
    await streamer.broadcast("StateChanged", {})
    assert ws.sent == []
