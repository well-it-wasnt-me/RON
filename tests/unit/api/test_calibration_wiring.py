"""Tests for calibration API wiring in the main DeskBot app.

The bug was that ``set_calibration_state()`` was only called from the
standalone CLI calibration server, never from the main app's
``_start_api()``.  This meant the web panel's calibration page always
got 503 on every endpoint because the module-level ``_state`` singleton
in ``calibration.py`` had ``servo_controller = None``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from robot.api.app import create_app
from robot.api.calibration import set_calibration_state
from robot.config import AppSettings
from robot.interfaces.servo import ServoController


@pytest.fixture
def mock_servo_controller() -> ServoController:
    """Create a real mock servo bus with 4 servos."""
    from robot.config import ServosConfig

    settings = ServosConfig(backend="mock")
    from robot.hardware.servos.factory import ServoControllerFactory

    return ServoControllerFactory(settings).build()


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(_env_file=None, env="testing", log_level="WARNING")


@pytest.fixture
def mock_display() -> MagicMock:
    d = MagicMock()
    d.width = 240
    d.height = 240
    d.show = AsyncMock()
    d.clear = AsyncMock()
    return d


@pytest.fixture
async def wired_app(settings, mock_servo_controller, mock_display):
    """Create a FastAPI app with calibration state wired to mock hardware."""
    app = create_app(settings=settings)
    set_calibration_state(
        servo_controller=mock_servo_controller,
        display=mock_display,
        settings=settings,
    )
    yield app
    # Cleanup: reset calibration state after test.
    set_calibration_state()


@pytest.fixture
async def client(wired_app):
    transport = ASGITransport(app=wired_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestCalibrationWiring:
    """Test that calibration endpoints work when wired via the main app."""

    async def test_list_servos(self, client: AsyncClient) -> None:
        """GET /api/v1/calibration/servos should return the servo list."""
        response = await client.get("/api/v1/calibration/servos")
        assert response.status_code == 200
        data = response.json()
        assert "servos" in data
        assert len(data["servos"]) == 4
        # Each servo should have name, angle, min_angle, max_angle, center_angle
        for servo in data["servos"]:
            assert "name" in servo
            assert "angle" in servo
            assert "min_angle" in servo
            assert "max_angle" in servo
            assert "center_angle" in servo

    async def test_move_servo(self, client: AsyncClient) -> None:
        """POST /api/v1/calibration/servos/{name}/move should move the servo."""
        response = await client.post(
            "/api/v1/calibration/servos/pan/move?angle=45.0&duration_s=0.2"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "pan"
        assert data["angle"] == 45.0

    async def test_move_servo_out_of_range(self, client: AsyncClient) -> None:
        """POST with an out-of-range angle should return 422."""
        response = await client.post(
            "/api/v1/calibration/servos/pan/move?angle=999.0&duration_s=0.2"
        )
        assert response.status_code == 422

    async def test_release_servo(self, client: AsyncClient) -> None:
        """POST /api/v1/calibration/servos/{name}/release should release the servo."""
        response = await client.post("/api/v1/calibration/servos/pan/release")
        assert response.status_code == 200
        data = response.json()
        assert data["released"] is True

    async def test_release_all(self, client: AsyncClient) -> None:
        """POST /api/v1/calibration/servos/release_all should release all servos."""
        response = await client.post("/api/v1/calibration/servos/release_all")
        assert response.status_code == 200
        data = response.json()
        assert data["released"] is True

    async def test_calibrate_servo(self, client: AsyncClient) -> None:
        """POST /api/v1/calibration/servos/calibrate/{name} should run calibration."""
        response = await client.post(
            "/api/v1/calibration/servos/calibrate/pan?include_limits=false"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["servo"] == "pan"
        assert len(data["sequence"]) == 1
        assert data["sequence"][0]["position"] == "centre"

    async def test_calibrate_servo_with_limits(self, client: AsyncClient) -> None:
        """POST with include_limits=true should visit min/centre/max/centre."""
        response = await client.post("/api/v1/calibration/servos/calibrate/pan?include_limits=true")
        assert response.status_code == 200
        data = response.json()
        assert data["servo"] == "pan"
        positions = [step["position"] for step in data["sequence"]]
        assert positions == ["min", "centre", "max", "centre"]

    async def test_get_display_config(self, client: AsyncClient) -> None:
        """GET /api/v1/calibration/display should return display config."""
        response = await client.get("/api/v1/calibration/display")
        assert response.status_code == 200
        data = response.json()
        assert "backend" in data
        assert "width" in data
        assert "height" in data

    async def test_show_test_pattern(self, client: AsyncClient) -> None:
        """POST /api/v1/calibration/display/test_pattern should show a pattern."""
        response = await client.post("/api/v1/calibration/display/test_pattern?pattern=gradient")
        assert response.status_code == 200
        data = response.json()
        assert data["pattern"] == "gradient"

    async def test_show_test_pattern_invalid(self, client: AsyncClient) -> None:
        """POST with an invalid pattern should return 400."""
        response = await client.post("/api/v1/calibration/display/test_pattern?pattern=bogus")
        assert response.status_code == 400

    async def test_clear_display(self, client: AsyncClient) -> None:
        """POST /api/v1/calibration/display/clear should clear the display."""
        response = await client.post("/api/v1/calibration/display/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["cleared"] is True


class TestCalibrationUnwired:
    """Test that calibration endpoints return 503 when not wired."""

    async def test_list_servos_503(self) -> None:
        """Without wiring, the calibration endpoints should return 503."""
        # Ensure calibration state is clean.
        set_calibration_state()
        settings = AppSettings(_env_file=None, env="testing")
        app = create_app(settings=settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/api/v1/calibration/servos")
            assert response.status_code == 503

    async def test_move_servo_503(self) -> None:
        set_calibration_state()
        settings = AppSettings(_env_file=None, env="testing")
        app = create_app(settings=settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/api/v1/calibration/servos/pan/move?angle=45.0")
            assert response.status_code == 503
