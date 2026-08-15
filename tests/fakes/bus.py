"""Recording event bus for tests."""

from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Any

from robot.interfaces.event_bus import EventHandler


class RecordingBus:
    """Captures every published event and records the subscriptions."""

    def __init__(self) -> None:
        self.published: list[Any] = []
        self.subscribed: dict[type, list[EventHandler]] = defaultdict(list)
        self._closed = False

    async def publish(self, event: object) -> None:
        if self._closed:
            return
        self.published.append(event)

    def subscribe(self, event_type: type, handler: EventHandler) -> None:
        self.subscribed[event_type].append(handler)

    def unsubscribe(self, event_type: type, handler: EventHandler) -> None:
        with contextlib.suppress(ValueError):
            self.subscribed[event_type].remove(handler)

    async def close(self) -> None:
        self._closed = True
        self.published.clear()
        self.subscribed.clear()

    def of_type[T](self, event_type: type[T]) -> list[T]:
        return [e for e in self.published if isinstance(e, event_type)]
