"""Tests for the FastAPI REST API."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from robot.api.app import create_app
from robot.api.state_bridge import StateBridge
from robot.behavior.state_machine import StateMachine
from robot.config import ApiConfig, AppSettings
from robot.events.bus import InMemoryEventBus


@pytest.fixture
def settings() -> AppSettings:
    """Create test settings."""
    return AppSettings(_env_file=None, env="testing", log_level="WARNING")


@pytest.fixture
def app(settings: AppSettings) -> object:
    """Create a test FastAPI app."""
    return create_app(settings=settings)


@pytest.fixture
async def client(app):
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def wired_app(settings: AppSettings) -> object:
    """Create a FastAPI app with a wired StateBridge."""
    app = create_app(settings=settings)
    bus = InMemoryEventBus()
    state_machine = StateMachine(bus=bus)
    bridge = StateBridge(
        bus=bus,
        state_machine=state_machine,
        conversation=None,
        tts=None,
        perception=None,
    )
    app.state.bridge = bridge
    return app


@pytest.fixture
async def wired_client(wired_app):
    """Create an async test client with a wired StateBridge."""
    transport = ASGITransport(app=wired_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_health_check(client: AsyncClient) -> None:
    """GET /api/v1/health returns ok."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


async def test_version_info(client: AsyncClient) -> None:
    """GET /api/v1/version returns version."""
    response = await client.get("/api/v1/version")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "1.0.0b0"
    assert data["name"] == "DeskBot API"


async def test_state_no_bridge(client: AsyncClient) -> None:
    """GET /api/v1/state returns unknown when no bridge is attached."""
    response = await client.get("/api/v1/state")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "unknown"


async def test_state_with_bridge(wired_client: AsyncClient) -> None:
    """GET /api/v1/state returns actual state when bridge is wired."""
    response = await wired_client.get("/api/v1/state")
    assert response.status_code == 200
    data = response.json()
    # State machine starts in BOOT state
    assert data["state"] == "boot"


async def test_config(client: AsyncClient) -> None:
    """GET /api/v1/config returns configuration."""
    response = await client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "env" in data
    assert data["env"] == "testing"


async def test_perception_no_bridge(client: AsyncClient) -> None:
    """GET /api/v1/perception returns disabled when no bridge is attached."""
    response = await client.get("/api/v1/perception")
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is False


async def test_audio_no_bridge(client: AsyncClient) -> None:
    """GET /api/v1/audio returns disabled when no bridge is attached."""
    response = await client.get("/api/v1/audio")
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is False


async def test_conversation_no_bridge(client: AsyncClient) -> None:
    """GET /api/v1/conversation returns disabled when no bridge is attached."""
    response = await client.get("/api/v1/conversation")
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is False


async def test_speak_no_bridge(client: AsyncClient) -> None:
    """POST /api/v1/speak returns error when no bridge is attached."""
    response = await client.post("/api/v1/speak", json={"text": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"


async def test_emotion_invalid(client: AsyncClient) -> None:
    """POST /api/v1/emotion with invalid emotion returns error."""
    response = await client.post("/api/v1/emotion", json={"emotion": "invalid_emotion"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"


async def test_state_transition_invalid(client: AsyncClient) -> None:
    """POST /api/v1/state with invalid state returns error."""
    response = await client.post("/api/v1/state", json={"state": "invalid_state"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"


async def test_state_transition_with_bridge(wired_client: AsyncClient) -> None:
    """POST /api/v1/state transitions the real state machine.

    State machine starts in BOOT. Transition from BOOT -> IDLE is allowed.
    """
    response = await wired_client.post("/api/v1/state", json={"state": "idle"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["state"] == "idle"


async def test_emotion_with_bridge(wired_client: AsyncClient) -> None:
    """POST /api/v1/emotion sets emotion via the event bus."""
    response = await wired_client.post(
        "/api/v1/emotion", json={"emotion": "happy", "intensity": 0.9}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["emotion"] == "happy"


async def test_create_app_with_settings() -> None:
    """create_app accepts custom settings."""
    settings = AppSettings(_env_file=None, env="testing", log_level="DEBUG")
    app = create_app(settings=settings)
    assert app.title == "DeskBot API"
    assert app.state.settings.env == "testing"


async def test_state_bridge_defaults() -> None:
    """StateBridge defaults to not ready."""
    bridge = StateBridge()
    assert not bridge.is_ready


async def test_state_bridge_ready() -> None:
    """StateBridge is ready when bus and state_machine are set."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    bridge = StateBridge(bus=bus, state_machine=sm)
    assert bridge.is_ready


async def test_api_config_defaults() -> None:
    """ApiConfig has sensible defaults."""
    cfg = ApiConfig()
    assert cfg.enabled is True
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000


async def test_api_config_env_override() -> None:
    """ApiConfig reads from env vars."""
    import os

    os.environ["DESKBOT_API__PORT"] = "9999"
    try:
        cfg = ApiConfig()
        assert cfg.port == 9999
    finally:
        del os.environ["DESKBOT_API__PORT"]
