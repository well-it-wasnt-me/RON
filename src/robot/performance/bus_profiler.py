"""Event bus throughput profiler - instruments publish latency and throughput.

Wraps :class:`InMemoryEventBus.publish()` to measure event processing time
for a configurable sample of events. Reports events/second, average
processing time, and maximum processing time.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from robot.logging import get_logger

if TYPE_CHECKING:
    from robot.events.bus import InMemoryEventBus

_log = get_logger("performance.bus_profiler")


class BusProfiler:
    """Instruments the event bus to measure throughput and latency.

    Parameters
    ----------
    bus:
        The event bus to instrument.
    sample_rate:
        Fraction of events to sample (0.0-1.0). A value of 0.1 means
        one in ten events is timed.
    window:
        Rolling window size for sampled events.
    enabled:
        When ``False``, all methods are no-ops.
    """

    def __init__(
        self,
        bus: InMemoryEventBus,
        sample_rate: float = 0.1,
        window: int = 500,
        enabled: bool = True,
    ) -> None:
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError("sample_rate must be in (0.0, 1.0]")
        if window <= 0:
            raise ValueError("window must be > 0")

        self._bus = bus
        self._sample_rate = sample_rate
        self._window = window
        self.enabled = enabled

        self._processing_times_ms: list[float] = []
        self._events_per_type: dict[str, int] = defaultdict(int)
        self._total_events: int = 0
        self._sampled_events: int = 0
        self._start_time: float = time.monotonic()
        self._original_publish = bus.publish
        self._installed = False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Install the profiling wrapper around ``bus.publish``."""
        if not self.enabled or self._installed:
            return

        original_publish = self._original_publish
        profiler = self

        async def _profiled_publish(event: object) -> None:
            # Always count the event
            profiler._total_events += 1
            event_type_name = type(event).__name__
            profiler._events_per_type[event_type_name] += 1

            # Decide whether to sample this event
            # Use a deterministic sampling based on counter to avoid
            # importing random for near-zero overhead.
            should_sample = (profiler._total_events % max(1, int(1.0 / profiler._sample_rate))) == 0

            if should_sample:
                start = time.monotonic()
                await original_publish(event)
                end = time.monotonic()
                processing_ms = (end - start) * 1000.0
                profiler._processing_times_ms.append(processing_ms)
                profiler._sampled_events += 1
                # Trim to rolling window
                if len(profiler._processing_times_ms) > profiler._window:
                    profiler._processing_times_ms = profiler._processing_times_ms[
                        -profiler._window :
                    ]
            else:
                await original_publish(event)

        self._bus.publish = _profiled_publish  # type: ignore[method-assign]
        self._installed = True
        _log.info("bus_profiler.started", sample_rate=self._sample_rate)

    def stop(self) -> None:
        """Restore the original ``bus.publish`` method."""
        if self._installed:
            self._bus.publish = self._original_publish  # type: ignore[method-assign]
            self._installed = False
            _log.info("bus_profiler.stopped")

    # ------------------------------------------------------------------ public API
    def stats(self) -> dict[str, Any]:
        """Return throughput and latency statistics."""
        elapsed = time.monotonic() - self._start_time
        events_per_sec = self._total_events / elapsed if elapsed > 0 else 0.0

        result: dict[str, Any] = {
            "total_events": self._total_events,
            "sampled_events": self._sampled_events,
            "sample_rate": self._sample_rate,
            "events_per_second": round(events_per_sec, 2),
            "elapsed_seconds": round(elapsed, 2),
            "events_by_type": dict(self._events_per_type),
        }

        if self._processing_times_ms:
            sorted_times = sorted(self._processing_times_ms)
            avg = sum(sorted_times) / len(sorted_times)
            result["avg_processing_time_ms"] = round(avg, 2)
            result["p50_processing_time_ms"] = round(self._percentile(sorted_times, 50), 2)
            result["p95_processing_time_ms"] = round(self._percentile(sorted_times, 95), 2)
            result["max_processing_time_ms"] = round(sorted_times[-1], 2)
        else:
            result["avg_processing_time_ms"] = 0.0
            result["p50_processing_time_ms"] = 0.0
            result["p95_processing_time_ms"] = 0.0
            result["max_processing_time_ms"] = 0.0

        return result

    def reset(self) -> None:
        """Clear all recorded data and reset the start time."""
        self._processing_times_ms.clear()
        self._events_per_type.clear()
        self._total_events = 0
        self._sampled_events = 0
        self._start_time = time.monotonic()

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


__all__ = ["BusProfiler"]
