"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from robot import __version__
from robot.api.schemas import HealthResponse, VersionResponse

router = APIRouter()


@router.get("/health", summary="Health check", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return overall health status including component degradation info.

    The response includes the overall status (``"ok"`` or ``"degraded"``)
    and per-component details when a :class:`DegradationRegistry` is
    available on the app state bridge.
    """
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is not None and hasattr(bridge, "degradation") and bridge.degradation is not None:
        return HealthResponse.model_validate(bridge.degradation.to_dict())
    return HealthResponse(status="ok")


@router.get("/version", summary="Version info", response_model=VersionResponse)
async def version_info() -> VersionResponse:
    """Return the API version."""
    return VersionResponse(version=__version__, name="DeskBot API")
