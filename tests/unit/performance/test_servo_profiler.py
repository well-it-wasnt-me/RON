"""Unit tests for ServoProfiler."""

from __future__ import annotations

import asyncio
import time

from robot.events.bus import InMemoryEventBus
from robot.events.events import ServoMoved
from robot.performance.servo_profiler import ServoLatency, ServoProfiler


class TestServoProfiler:
    """Tests for :class:`ServoProfiler`."""

    def test_record_command_and_event(self) -> None:
        bus = InMemoryEventBus()
        profiler = ServoProfiler(bus=bus, enabled=True)
        profiler.start()

        # Record a command then emit the event
        profiler.record_command("pan")
        asyncio.run(bus.publish(ServoMoved(name="pan", angle=45.0)))

        stats = profiler.stats()
        assert "pan" in stats
        assert stats["pan"]["count"] == 1.0
        assert stats["pan"]["avg_ms"] >= 0.0

        profiler.stop()

    def test_disabled_is_noop(self) -> None:
        bus = InMemoryEventBus()
        profiler = ServoProfiler(bus=bus, enabled=False)
        profiler.start()
        profiler.record_command("pan")
        # Should not crash and should not record anything
        stats = profiler.stats()
        assert stats == {}
        profiler.stop()

    def test_multiple_servos(self) -> None:
        bus = InMemoryEventBus()
        profiler = ServoProfiler(bus=bus, enabled=True)
        profiler.start()

        profiler.record_command("pan")
        asyncio.run(bus.publish(ServoMoved(name="pan", angle=10.0)))

        profiler.record_command("tilt")
        asyncio.run(bus.publish(ServoMoved(name="tilt", angle=20.0)))

        stats = profiler.stats()
        assert "pan" in stats
        assert "tilt" in stats

        profiler.stop()

    def test_rolling_window(self) -> None:
        bus = InMemoryEventBus()
        profiler = ServoProfiler(bus=bus, window=5, enabled=True)
        profiler.start()

        for i in range(10):
            profiler.record_command("pan")
            asyncio.run(bus.publish(ServoMoved(name="pan", angle=float(i))))

        stats = profiler.stats()
        # Only last 5 samples should be kept for statistics
        assert stats["pan"]["count"] == 5.0

        profiler.stop()

    def test_reset(self) -> None:
        bus = InMemoryEventBus()
        profiler = ServoProfiler(bus=bus, enabled=True)
        profiler.start()

        profiler.record_command("pan")
        asyncio.run(bus.publish(ServoMoved(name="pan", angle=0.0)))

        assert profiler.stats() != {}
        profiler.reset()
        assert profiler.stats() == {}

        profiler.stop()

    def test_stop_unsubscribes(self) -> None:
        bus = InMemoryEventBus()
        profiler = ServoProfiler(bus=bus, enabled=True)
        profiler.start()
        profiler.stop()
        # After stopping, new events should not be tracked
        profiler.record_command("pan")
        asyncio.run(bus.publish(ServoMoved(name="pan", angle=0.0)))
        stats = profiler.stats()
        assert stats == {}

    def test_event_without_command(self) -> None:
        """If a ServoMoved event arrives without a prior record_command, it's ignored."""
        bus = InMemoryEventBus()
        profiler = ServoProfiler(bus=bus, enabled=True)
        profiler.start()

        asyncio.run(bus.publish(ServoMoved(name="pan", angle=0.0)))
        stats = profiler.stats()
        assert stats == {}

        profiler.stop()


class TestServoLatency:
    """Tests for :class:`ServoLatency`."""

    def test_servo_latency_creation(self) -> None:
        latency = ServoLatency(
            name="pan",
            command_time=time.monotonic(),
            ack_time=time.monotonic(),
            latency_ms=5.0,
        )
        assert latency.name == "pan"
        assert latency.latency_ms == 5.0
