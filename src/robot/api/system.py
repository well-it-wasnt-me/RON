"""System info and log API routes for the web dashboard."""

from __future__ import annotations

import os
import platform
import time

from fastapi import APIRouter, Depends, Query, Request

from robot import __version__
from robot.api.schemas import (
    BluetoothResponse,
    LogsFiltersResponse,
    LogsResponse,
    OkResponse,
    SystemInfoResponse,
)
from robot.api.security import require_api_key
from robot.logging import get_ring_buffer

router = APIRouter(prefix="/system", tags=["system"])

# Process start time (set on first import).
_START_TIME = time.time()


def _split_csv(value: str | None) -> list[str] | None:
    """Split a comma-separated query value into a trimmed list (or ``None``)."""
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p] or None


@router.get("/info", summary="System information", response_model=SystemInfoResponse)
async def system_info(request: Request) -> SystemInfoResponse:
    """Return system information for the dashboard."""
    settings = getattr(request.app.state, "settings", None)
    uptime_s = time.time() - _START_TIME

    bridge = getattr(request.app.state, "bridge", None)
    health = {}
    if bridge is not None and hasattr(bridge, "degradation") and bridge.degradation is not None:
        health = bridge.degradation.to_dict()

    return SystemInfoResponse(
        hostname=platform.node(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 0,
        uptime_s=round(uptime_s, 1),
        uptime_human=_format_uptime(uptime_s),
        pid=os.getpid(),
        app_version=__version__,
        env=getattr(settings, "env", "production") if settings else "production",
        health=health,
    )


@router.get("/logs", summary="Recent log entries", response_model=LogsResponse)
async def system_logs(
    level: str | None = Query(
        default=None, description="Filter by level (DEBUG, INFO, WARNING, ERROR); 'ALL' = no filter"
    ),
    search: str | None = Query(
        default=None, description="Case-insensitive text search across event/logger/data"
    ),
    logger: str | None = Query(
        default=None, description="Case-insensitive substring filter on logger name"
    ),
    event: str | None = Query(
        default=None, description="Case-insensitive substring filter on event name"
    ),
    exclude: str | None = Query(
        default=None,
        description="Comma-separated event names to omit (e.g. 'DisplayUpdated,LookRequested')",
    ),
    since: float | None = Query(
        default=None,
        ge=0.0,
        description="Only entries with created_epoch >= this POSIX timestamp (for live tailing)",
    ),
    limit: int = Query(default=200, ge=1, le=500),
) -> LogsResponse:
    """Return recent log entries from the in-memory ring buffer.

    Filters apply server-side; the client can combine level/search/logger/
    event/exclude/since to drill into the buffer without pulling everything.
    """
    rb = get_ring_buffer()
    entries = rb.get_entries(
        level=level,
        search=search,
        logger=logger,
        event=event,
        exclude=_split_csv(exclude),
        since_epoch=since,
        limit=limit,
    )
    return LogsResponse(
        count=len(entries),
        entries=[
            {
                "timestamp": e.timestamp,
                "created_epoch": e.created_epoch,
                "level": e.level,
                "logger": e.logger_name,
                "event": e.event,
                "data": e.data,
            }
            for e in entries
        ],
    )


@router.get(
    "/logs/filters",
    summary="Distinct log filter values",
    response_model=LogsFiltersResponse,
)
async def system_logs_filters(request: Request) -> LogsFiltersResponse:
    """Return the distinct levels, logger names, and event names in the buffer.

    Also returns the server-configured ``noisy_events`` hide list
    (``LoggingConfig.noisy_events``) so the dashboard can mirror the same
    default in its Recent Events feed and ``/#/logs`` exclude toggle
    without hardcoding it in JS.
    """
    filters = get_ring_buffer().distinct_filters()
    settings = getattr(request.app.state, "settings", None)
    noisy = list(getattr(getattr(settings, "logging", None), "noisy_events", []) or [])
    return LogsFiltersResponse(
        levels=filters["levels"],
        loggers=filters["loggers"],
        events=filters["events"],
        noisy_events=noisy,
    )


@router.delete("/logs", summary="Clear log buffer", response_model=OkResponse)
async def clear_logs(_: None = Depends(require_api_key)) -> OkResponse:
    """Clear the in-memory log ring buffer."""
    get_ring_buffer().clear()
    return OkResponse(status="ok")


@router.get("/bluetooth", summary="Bluetooth status", response_model=BluetoothResponse)
async def bluetooth_status(request: Request) -> BluetoothResponse:
    """Return Bluetooth speaker status if available."""
    bridge = getattr(request.app.state, "bridge", None)
    audio = getattr(bridge, "audio", None) if bridge else None

    if audio is None:
        return BluetoothResponse(available=False)

    # Check if it's a BluetoothSpeaker
    audio_type = type(audio).__name__
    if "Bluetooth" not in audio_type:
        return BluetoothResponse(
            available=True,
            type=audio_type,
            is_bluetooth=False,
        )

    return BluetoothResponse(
        available=True,
        type=audio_type,
        is_bluetooth=True,
        connected=getattr(audio, "_connected", False),
        sink_name=getattr(audio, "_sink_name", ""),
        device_mac=getattr(audio, "device_mac", ""),
        device_name=getattr(audio, "device_name", ""),
        auto_connect=getattr(audio, "auto_connect", False),
        playing=getattr(audio, "_playing", False),
        sample_rate=getattr(audio, "sample_rate", None),
        channels=getattr(audio, "channels", None),
    )


def _format_uptime(seconds: float) -> str:
    """Format uptime seconds into a human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    if hours < 24:
        return f"{hours}h {mins}m"
    days = int(hours // 24)
    hrs = hours % 24
    return f"{days}d {hrs}h"


__all__ = ["router"]
