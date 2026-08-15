"""Deterministic clock for tests."""

from __future__ import annotations

from datetime import UTC, datetime

import anyio


class FakeClock:
    """Manually-advanced clock."""

    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, tzinfo=UTC)
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._mono += seconds
        from datetime import timedelta

        self._now = self._now + timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        # Tests don't actually need to wait; the clock is advanced explicitly.
        await anyio.sleep(0)
        self.advance(seconds)
