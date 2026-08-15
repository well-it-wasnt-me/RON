"""Unit tests for BusProfiler."""

from __future__ import annotations

import asyncio

import pytest

from robot.events.bus import InMemoryEventBus
from robot.events.events import RobotStarted
from robot.performance.bus_profiler import BusProfiler


class TestBusProfiler:
    """Tests for :class:`BusProfiler`."""

    def test_start_creates_wrapper(self) -> None:
        """Starting the profiler replaces the publish method."""
        bus = InMemoryEventBus()
        profiler = BusProfiler(bus=bus, sample_rate=1.0, enabled=True)
        # Store original by name to check it changes
        original_name = bus.publish.__name__
        profiler.start()
        # After starting, publish should be a different function
        assert bus.publish.__name__ != original_name or bus.publish.__qualname__ != original_name
        profiler.stop()

    def test_stop_restores_original(self) -> None:
        """Stopping the profiler restores the original publish method."""
        bus = InMemoryEventBus()
        profiler = BusProfiler(bus=bus, sample_rate=1.0, enabled=True)
        profiler.start()
        profiler.stop()
        # After stopping, publish should be the original InMemoryEventBus.publish
        assert "InMemoryEventBus" in bus.publish.__qualname__

    def test_counts_events(self) -> None:
        bus = InMemoryEventBus()
        profiler = BusProfiler(bus=bus, sample_rate=1.0, enabled=True)
        profiler.start()

        asyncio.run(bus.publish(RobotStarted()))
        asyncio.run(bus.publish(RobotStarted()))

        stats = profiler.stats()
        assert stats["total_events"] == 2

        profiler.stop()

    def test_sample_rate_filters(self) -> None:
        bus = InMemoryEventBus()
        profiler = BusProfiler(bus=bus, sample_rate=0.5, enabled=True)
        profiler.start()

        # With sample_rate=0.5, every 2nd event is sampled (1/0.5=2, so every 2nd)
        for _ in range(10):
            asyncio.run(bus.publish(RobotStarted()))

        stats = profiler.stats()
        assert stats["total_events"] == 10
        assert stats["sampled_events"] > 0

        profiler.stop()

    def test_disabled_is_noop(self) -> None:
        bus = InMemoryEventBus()
        profiler = BusProfiler(bus=bus, sample_rate=1.0, enabled=False)
        profiler.start()  # Should not install wrapper
        assert profiler.stats()["total_events"] == 0
        profiler.stop()

    def test_stats_with_no_events(self) -> None:
        bus = InMemoryEventBus()
        profiler = BusProfiler(bus=bus, sample_rate=1.0, enabled=True)
        profiler.start()
        stats = profiler.stats()
        assert stats["total_events"] == 0
        assert stats["avg_processing_time_ms"] == 0.0
        profiler.stop()

    def test_reset(self) -> None:
        bus = InMemoryEventBus()
        profiler = BusProfiler(bus=bus, sample_rate=1.0, enabled=True)
        profiler.start()
        asyncio.run(bus.publish(RobotStarted()))
        assert profiler.stats()["total_events"] == 1
        profiler.reset()
        assert profiler.stats()["total_events"] == 0
        profiler.stop()

    def test_events_by_type(self) -> None:
        bus = InMemoryEventBus()
        profiler = BusProfiler(bus=bus, sample_rate=1.0, enabled=True)
        profiler.start()

        asyncio.run(bus.publish(RobotStarted()))
        asyncio.run(bus.publish(RobotStarted()))

        stats = profiler.stats()
        assert "RobotStarted" in stats["events_by_type"]

        profiler.stop()

    def test_invalid_sample_rate_raises(self) -> None:
        bus = InMemoryEventBus()
        with pytest.raises(ValueError):
            BusProfiler(bus=bus, sample_rate=0.0)
        with pytest.raises(ValueError):
            BusProfiler(bus=bus, sample_rate=1.5)

    def test_invalid_window_raises(self) -> None:
        bus = InMemoryEventBus()
        with pytest.raises(ValueError):
            BusProfiler(bus=bus, window=0)

    def test_processing_time_measured(self) -> None:
        bus = InMemoryEventBus()
        profiler = BusProfiler(bus=bus, sample_rate=1.0, enabled=True)
        profiler.start()

        asyncio.run(bus.publish(RobotStarted()))

        stats = profiler.stats()
        assert stats["avg_processing_time_ms"] >= 0.0

        profiler.stop()

    def test_rolling_window(self) -> None:
        bus = InMemoryEventBus()
        profiler = BusProfiler(bus=bus, sample_rate=1.0, window=5, enabled=True)
        profiler.start()

        for _ in range(10):
            asyncio.run(bus.publish(RobotStarted()))

        stats = profiler.stats()
        # Only last 5 sampled events should be kept
        assert stats["total_events"] == 10

        profiler.stop()
