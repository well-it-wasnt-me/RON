"""Performance profiling API endpoints.

Exposes frame budget, servo latency, and event bus throughput statistics
as JSON endpoints under ``/api/v1/performance/``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Request

from robot.api.schemas import (
    BusProfilerStatsResponse,
    FrameStatsResponse,
    PerformanceSummaryResponse,
    ServoProfilerStatsResponse,
)

router = APIRouter()


@router.get(
    "/performance",
    summary="Combined performance summary",
    response_model=PerformanceSummaryResponse,
)
async def get_performance_summary(request: Request) -> PerformanceSummaryResponse:
    """Return a combined summary of all profiler data."""
    result: dict[str, Any] = {}

    frame_profiler = getattr(request.app.state, "frame_profiler", None)
    if frame_profiler is not None and getattr(frame_profiler, "enabled", False):
        result["frames"] = asdict(frame_profiler.stats())
    else:
        result["frames"] = {"enabled": False}

    servo_profiler = getattr(request.app.state, "servo_profiler", None)
    if servo_profiler is not None and getattr(servo_profiler, "enabled", False):
        result["servos"] = servo_profiler.stats()
    else:
        result["servos"] = {"enabled": False}

    bus_profiler = getattr(request.app.state, "bus_profiler", None)
    if bus_profiler is not None and getattr(bus_profiler, "enabled", False):
        result["bus"] = bus_profiler.stats()
    else:
        result["bus"] = {"enabled": False}

    return PerformanceSummaryResponse.model_validate(result)


@router.get("/performance/frames", summary="Frame budget stats", response_model=FrameStatsResponse)
async def get_frame_stats(request: Request) -> FrameStatsResponse:
    """Return frame budget performance statistics."""
    frame_profiler = getattr(request.app.state, "frame_profiler", None)
    if frame_profiler is None or not getattr(frame_profiler, "enabled", False):
        return FrameStatsResponse(enabled=False)
    return FrameStatsResponse.model_validate(dict(asdict(frame_profiler.stats())))


@router.get(
    "/performance/servos", summary="Servo latency stats", response_model=ServoProfilerStatsResponse
)
async def get_servo_stats(request: Request) -> ServoProfilerStatsResponse:
    """Return per-servo latency statistics."""
    servo_profiler = getattr(request.app.state, "servo_profiler", None)
    if servo_profiler is None or not getattr(servo_profiler, "enabled", False):
        return ServoProfilerStatsResponse(enabled=False)
    return ServoProfilerStatsResponse.model_validate(dict(servo_profiler.stats()))


@router.get(
    "/performance/bus", summary="Event bus throughput", response_model=BusProfilerStatsResponse
)
async def get_bus_stats(request: Request) -> BusProfilerStatsResponse:
    """Return event bus throughput and latency statistics."""
    bus_profiler = getattr(request.app.state, "bus_profiler", None)
    if bus_profiler is None or not getattr(bus_profiler, "enabled", False):
        return BusProfilerStatsResponse(enabled=False)
    return BusProfilerStatsResponse.model_validate(dict(bus_profiler.stats()))
