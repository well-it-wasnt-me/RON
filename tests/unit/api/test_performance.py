"""Unit tests for performance API endpoints."""

from __future__ import annotations

import httpx
import pytest

from robot.api.app import create_app
from robot.config import AppSettings
from robot.performance.frame_profiler import FrameProfiler


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(
        performance=__import__("robot.config", fromlist=["PerformanceConfig"]).PerformanceConfig(
            enabled=False
        )
    )


class TestPerformanceEndpoints:
    """Tests for ``/api/v1/performance/*`` endpoints."""

    @pytest.mark.asyncio
    async def test_combined_summary_disabled(self, settings: AppSettings) -> None:
        app = create_app(settings=settings)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/performance")
            assert response.status_code == 200
            data = response.json()
            assert "frames" in data
            assert "servos" in data
            assert "bus" in data

    @pytest.mark.asyncio
    async def test_frame_stats_endpoint_disabled(self, settings: AppSettings) -> None:
        app = create_app(settings=settings)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/performance/frames")
            assert response.status_code == 200
            data = response.json()
            assert data.get("enabled") is False

    @pytest.mark.asyncio
    async def test_servo_stats_endpoint_disabled(self, settings: AppSettings) -> None:
        app = create_app(settings=settings)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/performance/servos")
            assert response.status_code == 200
            data = response.json()
            assert data.get("enabled") is False

    @pytest.mark.asyncio
    async def test_bus_stats_endpoint_disabled(self, settings: AppSettings) -> None:
        app = create_app(settings=settings)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/performance/bus")
            assert response.status_code == 200
            data = response.json()
            assert data.get("enabled") is False

    @pytest.mark.asyncio
    async def test_frame_stats_with_enabled_profiler(self) -> None:
        settings = AppSettings()
        app = create_app(settings=settings)
        # Attach a frame profiler
        profiler = FrameProfiler(target_fps=30, enabled=True)
        app.state.frame_profiler = profiler
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/performance/frames")
            assert response.status_code == 200
            data = response.json()
            assert "total_frames" in data
