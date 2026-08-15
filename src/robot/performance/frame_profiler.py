"""Frame budget profiler - measures rendering performance and budget compliance.

Records per-frame timing (start -> end) and computes rolling-window statistics
(FPS, frame time percentiles, dropped frames). A frame is considered
*dropped* when its wall time exceeds the budget (``1000 / target_fps`` ms).

The profiler is designed to be called from the animation loop with
near-zero overhead when disabled: the caller should check
``profiler.enabled`` before calling :meth:`record_frame`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from robot.logging import get_logger

if TYPE_CHECKING:
    from robot.events.bus import InMemoryEventBus

_log = get_logger("performance.frame_profiler")


@dataclass(slots=True)
class FrameStats:
    """Snapshot of frame budget performance over a rolling window."""

    target_fps: int
    actual_fps: float
    avg_frame_time_ms: float
    p50_frame_time_ms: float
    p95_frame_time_ms: float
    p99_frame_time_ms: float
    dropped_frames: int
    total_frames: int


# ---------------------------------------------------------------------------
# Internal event published every N frames
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class FrameStatsReport:
    """Event payload published on the event bus every *report_interval* frames."""

    stats: FrameStats


# ---------------------------------------------------------------------------
# FrameProfiler
# ---------------------------------------------------------------------------
class FrameProfiler:
    """Measures frame rendering performance and reports budget compliance.

    Parameters
    ----------
    target_fps:
        The desired frame rate used to compute the per-frame budget.
    window:
        Rolling window size (number of recent frames kept for statistics).
    report_interval:
        Publish a :class:`FrameStatsReport` event every this many frames.
        Set to ``0`` to disable periodic reports.
    bus:
        Optional event bus for publishing :class:`FrameStatsReport` events.
    enabled:
        When ``False``, :meth:`record_frame` is a no-op.
    """

    def __init__(
        self,
        target_fps: int = 30,
        window: int = 100,
        report_interval: int = 100,
        bus: InMemoryEventBus | None = None,
        enabled: bool = True,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        if window <= 0:
            raise ValueError("window must be > 0")

        self._target_fps = target_fps
        self._window = window
        self._report_interval = report_interval
        self._bus = bus
        self.enabled = enabled

        self._frame_times_ms: list[float] = []
        self._total_frames: int = 0
        self._dropped_frames: int = 0
        self._budget_ms: float = 1000.0 / target_fps

    # ------------------------------------------------------------------ public API
    def record_frame(self, start: float, end: float) -> None:
        """Record the timing of a single frame.

        *start* and *end* must be monotonic timestamps
        (``time.monotonic()`` values). This method is a fast no-op when
        ``self.enabled`` is ``False``.
        """
        if not self.enabled:
            return

        frame_time_ms = (end - start) * 1000.0
        self._frame_times_ms.append(frame_time_ms)
        self._total_frames += 1

        # Trim to rolling window
        if len(self._frame_times_ms) > self._window:
            self._frame_times_ms = self._frame_times_ms[-self._window :]

        # Track dropped frames
        if frame_time_ms > self._budget_ms:
            self._dropped_frames += 1
            _log.warning(
                "frame.budget_exceeded",
                frame_time_ms=round(frame_time_ms, 2),
                budget_ms=round(self._budget_ms, 2),
            )

        # Periodic report
        if (
            self._report_interval > 0
            and self._total_frames % self._report_interval == 0
            and self._bus is not None
        ):
            import asyncio

            report = FrameStatsReport(stats=self.stats())
            try:
                loop = asyncio.get_running_loop()
                _task = loop.create_task(self._bus.publish(report))  # noqa: RUF006
            except RuntimeError:
                # No running loop: publish synchronously is not safe;
                # the report will simply be skipped.
                pass

    def stats(self) -> FrameStats:
        """Return a snapshot of current frame performance statistics."""
        if not self._frame_times_ms:
            return FrameStats(
                target_fps=self._target_fps,
                actual_fps=0.0,
                avg_frame_time_ms=0.0,
                p50_frame_time_ms=0.0,
                p95_frame_time_ms=0.0,
                p99_frame_time_ms=0.0,
                dropped_frames=0,
                total_frames=0,
            )

        times = sorted(self._frame_times_ms)
        n = len(times)
        avg = sum(times) / n
        actual_fps = 1000.0 / avg if avg > 0 else 0.0

        return FrameStats(
            target_fps=self._target_fps,
            actual_fps=round(actual_fps, 2),
            avg_frame_time_ms=round(avg, 2),
            p50_frame_time_ms=round(self._percentile(times, 50), 2),
            p95_frame_time_ms=round(self._percentile(times, 95), 2),
            p99_frame_time_ms=round(self._percentile(times, 99), 2),
            dropped_frames=self._dropped_frames,
            total_frames=self._total_frames,
        )

    def reset(self) -> None:
        """Clear all recorded data."""
        self._frame_times_ms.clear()
        self._total_frames = 0
        self._dropped_frames = 0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _percentile(sorted_data: list[float], pct: float) -> float:
        """Compute the *pct*-th percentile of sorted data."""
        if not sorted_data:
            return 0.0
        k = (pct / 100.0) * (len(sorted_data) - 1)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


__all__ = ["FrameProfiler", "FrameStats", "FrameStatsReport"]
