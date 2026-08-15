"""Clock abstraction so time can be controlled in tests."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    """Minimal time abstraction."""

    def now(self) -> datetime:
        """Return the current wall-clock time (UTC)."""

    def monotonic(self) -> float:
        """Return a strictly monotonic time in seconds."""

    async def sleep(self, seconds: float) -> None:
        """Sleep for the given number of seconds."""


class SystemClock:
    """Production clock backed by :mod:`time` and :mod:`anyio`."""

    def now(self) -> datetime:
        return datetime.now(tz=UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        import anyio

        await anyio.sleep(seconds)


__all__ = ["Clock", "SystemClock"]
