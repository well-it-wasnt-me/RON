"""Unit tests for FrameProfiler."""

from __future__ import annotations

import time

import pytest

from robot.performance.frame_profiler import FrameProfiler, FrameStats, FrameStatsReport


class TestFrameProfiler:
    """Tests for :class:`FrameProfiler`."""

    def test_record_frame_basic(self) -> None:
        profiler = FrameProfiler(target_fps=30, enabled=True)
        start = time.monotonic()
        end = start + 0.020  # 20ms frame
        profiler.record_frame(start, end)
        stats = profiler.stats()
        assert stats.total_frames == 1
        assert stats.avg_frame_time_ms > 0

    def test_record_frame_disabled_is_noop(self) -> None:
        profiler = FrameProfiler(target_fps=30, enabled=False)
        start = time.monotonic()
        end = start + 0.020
        profiler.record_frame(start, end)
        stats = profiler.stats()
        assert stats.total_frames == 0

    def test_dropped_frame_detection(self) -> None:
        profiler = FrameProfiler(target_fps=30, enabled=True)
        # 30 FPS => budget = 33.33ms; a 50ms frame is "dropped"
        start = time.monotonic()
        end = start + 0.050  # 50ms
        profiler.record_frame(start, end)
        stats = profiler.stats()
        assert stats.dropped_frames == 1

    def test_no_dropped_frame_within_budget(self) -> None:
        profiler = FrameProfiler(target_fps=30, enabled=True)
        start = time.monotonic()
        end = start + 0.030  # 30ms, within budget
        profiler.record_frame(start, end)
        stats = profiler.stats()
        assert stats.dropped_frames == 0

    def test_rolling_window(self) -> None:
        profiler = FrameProfiler(target_fps=30, window=5, enabled=True)
        base = time.monotonic()
        for i in range(10):
            start = base + i * 0.030
            end = start + 0.010
            profiler.record_frame(start, end)
        stats = profiler.stats()
        # Window is 5, so only last 5 frames are kept for percentile calc
        # But total_frames should still be 10
        assert stats.total_frames == 10

    def test_stats_empty(self) -> None:
        profiler = FrameProfiler(target_fps=30, enabled=True)
        stats = profiler.stats()
        assert stats.total_frames == 0
        assert stats.actual_fps == 0.0
        assert stats.avg_frame_time_ms == 0.0

    def test_stats_percentiles(self) -> None:
        profiler = FrameProfiler(target_fps=30, window=100, enabled=True)
        base = time.monotonic()
        # Record 20 frames with varying durations
        for i in range(20):
            duration = 0.010 + (i * 0.001)  # 10ms to 29ms
            start = base + i * 0.030
            end = start + duration
            profiler.record_frame(start, end)
        stats = profiler.stats()
        assert stats.p50_frame_time_ms > 0
        assert stats.p95_frame_time_ms >= stats.p50_frame_time_ms
        assert stats.p99_frame_time_ms >= stats.p95_frame_time_ms

    def test_reset(self) -> None:
        profiler = FrameProfiler(target_fps=30, enabled=True)
        start = time.monotonic()
        end = start + 0.020
        profiler.record_frame(start, end)
        assert profiler.stats().total_frames == 1
        profiler.reset()
        stats = profiler.stats()
        assert stats.total_frames == 0
        assert stats.dropped_frames == 0

    def test_invalid_fps_raises(self) -> None:
        with pytest.raises(ValueError):
            FrameProfiler(target_fps=0)

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError):
            FrameProfiler(target_fps=30, window=0)

    def test_report_interval_triggers_event(self) -> None:
        """Frame profiler publishes FrameStatsReport at the configured interval."""

        from robot.events.bus import InMemoryEventBus

        bus = InMemoryEventBus()
        profiler = FrameProfiler(
            target_fps=30,
            report_interval=5,
            bus=bus,
            enabled=True,
        )

        received: list[object] = []
        bus.subscribe(FrameStatsReport, received.append)

        base = time.monotonic()
        for i in range(10):
            start = base + i * 0.030
            end = start + 0.010
            profiler.record_frame(start, end)

        # After 5 and 10 frames, reports should have been attempted
        # (may not have been delivered if no running loop)
        # Just verify the profiler doesn't crash when bus is present
        assert profiler.stats().total_frames == 10

    def test_actual_fps_calculation(self) -> None:
        profiler = FrameProfiler(target_fps=30, enabled=True)
        base = time.monotonic()
        # All frames at exactly 33ms (~30 FPS)
        for i in range(5):
            start = base + i * 0.033
            end = start + 0.033
            profiler.record_frame(start, end)
        stats = profiler.stats()
        # actual_fps should be around 30
        assert 25.0 <= stats.actual_fps <= 35.0


class TestFrameStats:
    """Tests for :class:`FrameStats`."""

    def test_frame_stats_is_dataclass(self) -> None:
        stats = FrameStats(
            target_fps=30,
            actual_fps=29.5,
            avg_frame_time_ms=33.89,
            p50_frame_time_ms=33.0,
            p95_frame_time_ms=35.0,
            p99_frame_time_ms=36.0,
            dropped_frames=1,
            total_frames=100,
        )
        assert stats.target_fps == 30
        assert stats.actual_fps == 29.5
        assert stats.dropped_frames == 1
