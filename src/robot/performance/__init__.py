"""Lightweight performance profiling for DeskBot.

Provides frame budget monitoring, servo latency tracking, and event bus
throughput instrumentation. All profilers are *opt-in* and have near-zero
overhead when :class:`PerformanceConfig`.enabled is ``False``.
"""

from __future__ import annotations

from robot.performance.bus_profiler import BusProfiler
from robot.performance.frame_profiler import FrameProfiler, FrameStats
from robot.performance.servo_profiler import ServoLatency, ServoProfiler

__all__ = [
    "BusProfiler",
    "FrameProfiler",
    "FrameStats",
    "ServoLatency",
    "ServoProfiler",
]
