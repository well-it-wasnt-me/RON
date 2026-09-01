"""WebSocket endpoint for real-time event streaming.

The WebSocket at ``/api/v1/ws/events`` streams every event published on
the :class:`InMemoryEventBus` as JSON to connected clients. This enables
real-time dashboards, browser-based control panels, and external
integrations (Home Assistant, Node-RED, etc.) to observe robot state
changes, emotions, face detections, wake words, and more - all without
polling.

Message format
~~~~~~~~~~~~~~
Each message is a JSON object with two keys:

* ``type`` - the event class name (e.g. ``"StateChanged"``,
  ``"EmotionChanged"``, ``"FaceDetected"``).
* ``data`` - the event payload as a plain dict (dataclass fields).

Example::

    {"type": "StateChanged", "data": {"previous": "idle", "current": "curious"}}
    {"type": "EmotionChanged", "data": {"previous": "neutral", "current": "happy", "intensity": 0.8}}
    {"type": "FaceDetected", "data": {"x": 0.45, "y": 0.32, "confidence": 0.92}}

Per-connection filtering
~~~~~~~~~~~~~~~~~~~~~~~~
The event bus itself stays unfiltered (every subscriber still receives
every event); only **per-client delivery** is filtered, so a browser can
opt out of the high-frequency firehose without affecting other clients.

Filters are set in two ways:

1. **Query params on connect**::

       ws://<host>:8000/api/v1/ws/events?exclude=DisplayUpdated,LookRequested
       ws://<host>:8000/api/v1/ws/events?include=StateChanged,EmotionChanged

   ``include`` wins over ``exclude`` when both are given: only the listed
   event types are delivered.

2. **Runtime update message** (after connect, alongside ``ping``)::

       {"filter": {"include": ["StateChanged"], "exclude": ["DisplayUpdated"]}}

   Either field may be omitted or null to clear it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from dataclasses import dataclass, fields
from datetime import date, datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from robot.logging import get_logger

_log = get_logger("api.ws")

router = APIRouter()


def _event_to_dict(event: object) -> dict[str, Any]:
    """Convert an event dataclass to a JSON-serializable dict."""
    if not dataclasses.is_dataclass(event):
        return {"repr": repr(event)}
    result: dict[str, Any] = {}
    for f in fields(event):
        value = getattr(event, f.name)
        if isinstance(value, Enum):
            result[f.name] = value.value
        else:
            result[f.name] = value
    return result


def _split_csv(value: str | None) -> set[str] | None:
    """Split a comma-separated value into a set (or ``None`` if empty)."""
    if value is None:
        return None
    parts = {p.strip() for p in value.split(",")}
    parts.discard("")
    return parts or None


@dataclass
class EventFilter:
    """Per-connection event type filter.

    ``include`` and ``exclude`` are sets of event type names (the
    ``type(event).__name__`` value). When ``include`` is set, only those
    types are delivered (``exclude`` is ignored). Otherwise everything
    except the ``exclude`` set is delivered. Both ``None`` means deliver
    everything.
    """

    include: set[str] | None = None
    exclude: set[str] | None = None

    def matches(self, event_type: str) -> bool:
        """Return whether *event_type* passes this filter."""
        if self.include is not None:
            return event_type in self.include
        if self.exclude is not None:
            return event_type not in self.exclude
        return True


class EventStreamer:
    """Bridge between the event bus and WebSocket clients.

    Subscribes to all events on the bus and forwards them as JSON
    messages to every connected WebSocket, honouring each connection's
    :class:`EventFilter`.
    """

    def __init__(self) -> None:
        # (websocket, filter) tuples so delivery can be filtered per client.
        self._connections: list[tuple[WebSocket, EventFilter]] = []

    async def connect(self, websocket: WebSocket, event_filter: EventFilter | None = None) -> None:
        """Accept a WebSocket connection and start streaming events."""
        await websocket.accept()
        self._connections.append((websocket, event_filter or EventFilter()))
        _log.info("ws.client_connected", clients=len(self._connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        with contextlib.suppress(ValueError):
            self._connections = [(ws, f) for ws, f in self._connections if ws is not websocket]
        _log.info("ws.client_disconnected", clients=len(self._connections))

    def set_filter(self, websocket: WebSocket, event_filter: EventFilter) -> None:
        """Update the filter for an existing connection (runtime update)."""
        self._connections = [
            (ws, event_filter if ws is websocket else f) for ws, f in self._connections
        ]

    async def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Send a JSON message to all connected clients whose filter matches."""
        message = json.dumps(
            {"type": event_type, "data": data},
            default=_json_default,
        )
        stale: list[WebSocket] = []
        for ws, event_filter in self._connections:
            if not event_filter.matches(event_type):
                continue
            try:
                await ws.send_text(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)

    async def on_event(self, event: object) -> None:
        """Event handler subscribed to the bus."""
        event_type = type(event).__name__
        data = _event_to_dict(event)
        await self.broadcast(event_type, data)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


def _json_default(value: object) -> object:
    """Convert common event payload types to JSON-compatible values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


# Module-level singleton for the event streamer.
_streamer = EventStreamer()


def get_streamer() -> EventStreamer:
    """Return the module-level :class:`EventStreamer` singleton."""
    return _streamer


def _parse_filter_message(payload: str) -> EventFilter | None:
    """Parse a runtime ``{"filter": {...}}`` update message.

    Returns ``None`` if *payload* is not a filter-update message.
    """
    try:
        obj = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or "filter" not in obj:
        return None
    spec = obj["filter"]
    if not isinstance(spec, dict):
        return None
    include_raw = spec.get("include")
    exclude_raw = spec.get("exclude")
    include = {str(t) for t in include_raw} if isinstance(include_raw, list) else None
    exclude = {str(t) for t in exclude_raw} if isinstance(exclude_raw, list) else None
    return EventFilter(include=include, exclude=exclude)


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    """Stream events from the robot's event bus as JSON.

    Connect a WebSocket client to ``ws://<host>:8000/api/v1/ws/events``
    to receive a live feed of state changes, emotions, face detections,
    wake words, and more. Optional ``include``/``exclude`` query params
    filter which event types are delivered to this connection; a runtime
    ``{"filter": {...}}`` message updates the filter after connect.
    """
    streamer = get_streamer()

    # Parse the initial filter from query params.
    include = _split_csv(websocket.query_params.get("include"))
    exclude = _split_csv(websocket.query_params.get("exclude"))
    initial_filter = EventFilter(include=include, exclude=exclude)

    await streamer.connect(websocket, initial_filter)
    try:
        # Keep the connection alive. The client can send "ping" messages
        # (echoed back), or a {"filter": {...}} message to update its
        # event-type filter at runtime.
        while True:
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            if data.strip() == "ping":
                await websocket.send_text("pong")
                continue
            new_filter = _parse_filter_message(data)
            if new_filter is not None:
                streamer.set_filter(websocket, new_filter)
                _log.info(
                    "ws.filter_updated",
                    include=sorted(new_filter.include) if new_filter.include else None,
                    exclude=sorted(new_filter.exclude) if new_filter.exclude else None,
                )
    finally:
        streamer.disconnect(websocket)


__all__ = ["EventFilter", "EventStreamer", "get_streamer", "router"]
