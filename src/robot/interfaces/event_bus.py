"""Event bus interface."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeAlias, runtime_checkable

EventHandler: TypeAlias = Callable[[Any], Awaitable[None] | None]


@runtime_checkable
class EventBus(Protocol):
    """Asynchronous pub/sub event bus.

    Handlers may be sync or async; if a handler raises, the error is logged
    but does not stop other handlers from running.
    """

    async def publish(self, event: object) -> None:
        """Publish an event to every subscriber."""

    def subscribe(self, event_type: type, handler: EventHandler) -> None:
        """Subscribe to events of a specific type."""

    def unsubscribe(self, event_type: type, handler: EventHandler) -> None:
        """Unsubscribe a previously-registered handler."""

    async def close(self) -> None:
        """Cancel any background tasks and release resources."""
