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
"""

from __future__ import annotations

import contextlib
import json
import dataclasses
from dataclasses import fields
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
    for f in fields(event):  # type: ignore[arg-type]
        value = getattr(event, f.name)
        if isinstance(value, Enum):
            result[f.name] = value.value
        else:
            result[f.name] = value
    return result


class EventStreamer:
    """Bridge between the event bus and WebSocket clients.

    Subscribes to all events on the bus and forwards them as JSON
    messages to every connected WebSocket.
    """

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a WebSocket connection and start streaming events."""
        await websocket.accept()
        self._connections.append(websocket)
        _log.info("ws.client_connected", clients=len(self._connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        with contextlib.suppress(ValueError):
            self._connections.remove(websocket)
        _log.info("ws.client_disconnected", clients=len(self._connections))

    async def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Send a JSON message to all connected clients."""
        message = json.dumps(
            {"type": event_type, "data": data},
            default=_json_default,
        )
        stale: list[WebSocket] = []
        for ws in self._connections:
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


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    """Stream all events from the robot's event bus as JSON.

    Connect a WebSocket client to ``ws://<host>:8000/api/v1/ws/events``
    to receive a live feed of state changes, emotions, face detections,
    wake words, and more.
    """
    streamer = get_streamer()
    await streamer.connect(websocket)
    try:
        # Keep the connection alive. The client can send "ping" messages
        # and we echo them back; this also detects disconnections.
        while True:
            try:
                data = await websocket.receive_text()
                if data.strip() == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
    finally:
        streamer.disconnect(websocket)


__all__ = ["EventStreamer", "get_streamer", "router"]
