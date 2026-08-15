"""Servo latency profiler - measures end-to-end servo command latency.

Records the time from when a :class:`ServoMoved` event is published to
when the servo acknowledges the move, providing per-servo latency
statistics (avg, p50, p95).

For real servos (GPIO/PCA9685), latency includes physical movement time.
For mock servos, latency should be near-zero.

The profiler subscribes to ``ServoMoved`` events on the bus and optionally
measures the time between the ``move_to()`` call and the event emission.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from robot.logging import get_logger

if TYPE_CHECKING:
    from robot.events.bus import InMemoryEventBus

_log = get_logger("performance.servo_profiler")


@dataclass(slots=True)
class ServoLatency:
    """A single servo latency measurement."""

    name: str
    command_time: float
    ack_time: float
    latency_ms: float


class ServoProfiler:
    """Measures end-to-end latency from servo command to completion.

    The profiler intercepts :class:`ServoMoved` events on the bus and
    records the time between when a servo move was initiated and when it
    was acknowledged. Since the event bus is synchronous within the
    publisher's task, the *ack_time* is set to the event arrival time.

    For more accurate latency tracking, callers should record the
    *command_time* via :meth:`record_command` just before calling
    ``servo.move_to()``; the ``ServoMoved`` event handler will then
    compute the delta.
    """

    def __init__(
        self,
        bus: InMemoryEventBus,
        window: int = 100,
        enabled: bool = True,
    ) -> None:
        self._bus = bus
        self._window = window
        self.enabled = enabled

        # Per-servo rolling latency measurements (ms)
        self._latencies_ms: dict[str, list[float]] = defaultdict(list)
        # Pending command timestamps (monotonic) keyed by servo name
        self._pending_commands: dict[str, float] = {}
        self._total_commands: int = 0
        self._handler_id: object | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Subscribe to :class:`ServoMoved` events on the bus."""
        if not self.enabled:
            return
        from robot.events.events import ServoMoved

        self._handler_id = _ServoMovedHandler(self)
        self._bus.subscribe(ServoMoved, self._handler_id)
        _log.info("servo_profiler.started")

    def stop(self) -> None:
        """Unsubscribe from the bus."""
        from robot.events.events import ServoMoved

        if self._handler_id is not None:
            self._bus.unsubscribe(ServoMoved, self._handler_id)  # type: ignore[arg-type]
            self._handler_id = None
        _log.info("servo_profiler.stopped")

    # ------------------------------------------------------------------ public API
    def record_command(self, name: str) -> None:
        """Record the time just before a ``move_to()`` call.

        Call this *immediately before* ``servo.move_to()`` so that the
        profiler can compute end-to-end latency when the corresponding
        ``ServoMoved`` event arrives.
        """
        if not self.enabled:
            return
        self._pending_commands[name] = time.monotonic()

    def handle_servo_moved(self, name: str) -> None:
        """Called when a :class:`ServoMoved` event arrives on the bus."""
        if not self.enabled:
            return
        now = time.monotonic()
        command_time = self._pending_commands.pop(name, None)
        if command_time is not None:
            latency_ms = (now - command_time) * 1000.0
            self._latencies_ms[name].append(latency_ms)
            # Trim to rolling window
            if len(self._latencies_ms[name]) > self._window:
                self._latencies_ms[name] = self._latencies_ms[name][-self._window :]
            self._total_commands += 1

    def stats(self) -> dict[str, dict[str, float]]:
        """Return per-servo latency statistics.

        Returns a dict mapping servo name to ``{avg, p50, p95, count}``.
        """
        result: dict[str, dict[str, float]] = {}
        for name, latencies in self._latencies_ms.items():
            if not latencies:
                continue
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            avg = sum(sorted_lat) / n
            result[name] = {
                "avg_ms": round(avg, 2),
                "p50_ms": round(self._percentile(sorted_lat, 50), 2),
                "p95_ms": round(self._percentile(sorted_lat, 95), 2),
                "count": float(n),
            }
        return result

    def reset(self) -> None:
        """Clear all recorded data."""
        self._latencies_ms.clear()
        self._pending_commands.clear()
        self._total_commands = 0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _percentile(sorted_data: list[float], pct: float) -> float:
        if not sorted_data:
            return 0.0
        k = (pct / 100.0) * (len(sorted_data) - 1)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class _ServoMovedHandler:
    """Callable wrapper so the profiler can be subscribed to ServoMoved events."""

    __slots__ = ("_profiler",)

    def __init__(self, profiler: ServoProfiler) -> None:
        self._profiler = profiler

    async def __call__(self, event: object) -> None:
        name = getattr(event, "name", None)
        if isinstance(name, str):
            self._profiler.handle_servo_moved(name)

    @property
    def __name__(self) -> str:
        return "servo_profiler_handler"


__all__ = ["ServoLatency", "ServoProfiler"]
