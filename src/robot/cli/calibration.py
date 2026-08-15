"""Run the standalone DeskBot calibration server."""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from robot.api.calibration import router as calibration_router, set_calibration_state
from robot.config import load_settings
from robot.hardware.servos.factory import ServoControllerFactory

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deskbot-calibration",
        description="Run the standalone DeskBot servo calibration server.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    return parser.parse_args()


def create_calibration_app() -> FastAPI:
    """Create a minimal API that owns only the calibration hardware."""
    settings = load_settings()
    if settings.servos.backend != "gpio":
        raise RuntimeError(
            "deskbot-calibration requires DESKBOT_SERVOS__BACKEND=gpio; "
            f"current backend is {settings.servos.backend!r}"
        )

    servo_controller = ServoControllerFactory(settings.servos).build()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        set_calibration_state(servo_controller=servo_controller, settings=settings)
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                await servo_controller.close()
            set_calibration_state()

    app = FastAPI(
        title="DeskBot Calibration",
        version="1.0.0",
        description="Standalone servo calibration and hardware tuning API.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Expose the calibration router at both paths. The standalone UI uses the
    # short path, while existing/cached DeskBot pages use /api/v1.
    app.include_router(calibration_router)
    app.include_router(calibration_router, prefix="/api/v1")

    @app.websocket("/api/v1/ws/events")
    async def legacy_events(websocket: WebSocket) -> None:
        """Accept the normal UI event socket without requiring DeskBot itself."""
        await websocket.accept()
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass

    @app.middleware("http")
    async def no_cache(request: Request, call_next: Callable[..., Any]) -> Any:
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.endswith(".html"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    calibration_ui = _PACKAGE_ROOT / "web" / "calibration"
    if calibration_ui.is_dir():
        app.mount("/", StaticFiles(directory=str(calibration_ui), html=True), name="calibration-ui")

    return app


def main() -> int:
    args = _parse_args()
    uvicorn.run(create_calibration_app(), host=args.host, port=args.port)
    return 0


__all__ = ["create_calibration_app", "main"]
