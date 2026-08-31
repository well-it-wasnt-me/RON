"""In-memory implementation of :class:`EventBus`.

Handlers run sequentially on the publisher's task. This is intentional - the
robot is single-user and the predictable ordering keeps behaviour tests
deterministic.

Handlers are classified as either **critical** or **non-critical**:

* Critical handlers propagate their exceptions to the publisher, allowing
  callers to catch and recover.  These are for state-machine transitions,
  conversation pipeline steps, and any handler whose failure would leave
  the robot in an inconsistent state.

* Non-critical handlers have their exceptions logged and isolated.  These
  are for telemetry, profiling, learning, and other observers whose failure
  must not interrupt the main event flow.

Use :meth:`subscribe` for non-critical handlers (the default) and
:meth:`subscribe_critical` for handlers whose failure should propagate.
"""

from __future__ import annotations

import asyncio
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from robot.interfaces.event_bus import EventHandler
from robot.logging import get_logger

_log = get_logger("events.bus")


@dataclass(slots=True, frozen=True)
class HandlerFailedError(Exception):
    """Raised when a critical event handler fails during publish."""

    event_type: str
    handler: str
    original: BaseException


@dataclass(slots=True)
class _Subscription:
    """Internal representation of an event subscription."""

    handler: Any
    critical: bool = False


class InMemoryEventBus:
    """An async pub/sub bus for in-process events with handler priority.

    Critical handlers have their exceptions propagated; non-critical
    handlers have their exceptions logged and isolated.
    """

    def __init__(self) -> None:
        self._subscribers: dict[type, list[_Subscription]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        self._closed = False

    async def publish(self, event: object) -> None:
        """Publish an event to all subscribers."""
        if self._closed:
            return
        event_type = type(event)
        async with self._lock:
            subs = list(self._subscribers.get(event_type, []))
            if event_type is not object:
                subs.extend(self._subscribers.get(object, []))

        critical_subs = [s for s in subs if s.critical]
        non_critical_subs = [s for s in subs if not s.critical]

        critical_failures: list[HandlerFailedError] = []
        for sub in critical_subs:
            handler = sub.handler
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                hf = HandlerFailedError(
                    event_type=event_type.__name__,
                    handler=getattr(handler, "__qualname__", repr(handler)),
                    original=exc,
                )
                critical_failures.append(hf)
                _log.exception(
                    "events.critical_handler_failed",
                    event_type=event_type.__name__,
                    handler=hf.handler,
                )

        for sub in non_critical_subs:
            handler = sub.handler
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                _log.exception(
                    "events.handler_failed",
                    event_type=event_type.__name__,
                    handler=getattr(handler, "__qualname__", repr(handler)),
                )

        if critical_failures:
            if len(critical_failures) > 1:
                _log.warning(
                    "events.suppressed_critical_failures",
                    count=len(critical_failures) - 1,
                )
            raise critical_failures[0]

    def subscribe(self, event_type: type, handler: EventHandler) -> None:
        """Subscribe a non-critical handler for events of ``event_type``."""
        sub = _Subscription(handler=handler, critical=False)
        with self._sync_lock:
            if sub in self._subscribers[event_type]:
                return
            self._subscribers[event_type].append(sub)
        _log.debug("events.subscribed", event_type=event_type.__name__, handler=handler.__name__)

    def subscribe_critical(self, event_type: type, handler: EventHandler) -> None:
        """Subscribe a critical handler for events of ``event_type``."""
        sub = _Subscription(handler=handler, critical=True)
        with self._sync_lock:
            self._subscribers[event_type].append(sub)
        _log.debug(
            "events.subscribed_critical",
            event_type=event_type.__name__,
            handler=handler.__name__,
        )

    def unsubscribe(self, event_type: type, handler: EventHandler) -> None:
        """Unsubscribe a previously-registered handler."""
        with self._sync_lock:
            self._subscribers[event_type] = [
                s for s in self._subscribers[event_type] if s.handler != handler
            ]

    async def close(self) -> None:
        self._closed = True
        self._subscribers.clear()

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._subscribers.values())
        return f"InMemoryEventBus(subscribers={total}, closed={self._closed})"
