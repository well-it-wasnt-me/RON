"""FastAPI application factory for DeskBot's REST API.

The API exposes health, configuration, state, and command endpoints
so that external tools (dashboards, Home Assistant, phone apps) can
monitor and control the robot over the network.

Usage::

    from robot.api import create_app

    app = create_app()

Run with::

    uvicorn robot.api.app:create_app --factory --host 0.0.0.0 --port 8000

Or programmatically::

    import uvicorn
    from robot.api import create_app

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from robot import __version__
from robot.api.calibration import router as calibration_router
from robot.api.config_validation import router as config_validation_router
from robot.api.learning import router as learning_router
from robot.api.performance import router as performance_router
from robot.api.preferences import router as preferences_router
from robot.api.routes import commands, conversations, health, state
from robot.api.settings import router as settings_router
from robot.api.state_bridge import StateBridge
from robot.api.system import router as system_router
from robot.api.ws import router as ws_router
from robot.config import AppSettings, load_settings

_api_state: dict[str, FastAPI | None] = {"app": None}

# Root package directory used to locate static assets.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    settings:
        Application settings. When ``None``, loaded from environment /
        .env via :func:`load_settings`.
    """
    settings = settings or load_settings()

    app = FastAPI(
        title="DeskBot API",
        version=__version__,
        description="REST API for monitoring and controlling DeskBot.",
    )

    # Allow cross-origin requests from any dashboard.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store settings and an empty state bridge on the app state.
    app.state.settings = settings
    app.state.deskbot = None  # Set later by DeskBotApp
    app.state.bridge = StateBridge()  # Populated by DeskBotApp

    # Register routers.
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(state.router, prefix="/api/v1", tags=["state"])
    app.include_router(commands.router, prefix="/api/v1", tags=["commands"])
    app.include_router(conversations.router, prefix="/api/v1", tags=["conversations"])
    app.include_router(calibration_router, prefix="/api/v1", tags=["calibration"])
    app.include_router(preferences_router, prefix="/api/v1", tags=["preferences"])
    app.include_router(learning_router, prefix="/api/v1", tags=["learning"])
    app.include_router(config_validation_router, prefix="/api/v1", tags=["config"])
    app.include_router(performance_router, prefix="/api/v1", tags=["performance"])
    app.include_router(settings_router, prefix="/api/v1", tags=["settings"])
    app.include_router(system_router, prefix="/api/v1", tags=["system"])
    app.include_router(ws_router, prefix="/api/v1")

    # Serve the calibration UI from web/calibration/ if it exists.
    _mount_static(app, "/calibration", _PACKAGE_ROOT / "web" / "calibration", "calibration")

    # Serve the config validator UI from web/config-validator/ if it exists.
    _mount_static(app, "/config", _PACKAGE_ROOT / "web" / "config-validator", "config-validator")

    # Serve the settings UI from web/settings/ if it exists.
    _mount_static(app, "/settings", _PACKAGE_ROOT / "web" / "settings", "settings")
    # Serve the learning dashboard from web/learning/ if it exists.
    _mount_static(app, "/learning", _PACKAGE_ROOT / "web" / "learning", "learning")

    # Serve the main web dashboard from web/ if it exists.
    _mount_static(app, "", _PACKAGE_ROOT / "web", "web")

    _api_state["app"] = app
    return app


def _mount_static(
    app: FastAPI,
    path: str,
    directory: Path,
    name: str,
) -> None:
    """Mount a static-files directory at *path* if it exists."""
    if directory.is_dir():
        app.mount(path, StaticFiles(directory=str(directory), html=True), name=name)


def get_app() -> FastAPI:
    """Return the current FastAPI application instance."""
    if _api_state["app"] is None:
        return create_app()
    return _api_state["app"]


__all__ = ["create_app", "get_app"]
