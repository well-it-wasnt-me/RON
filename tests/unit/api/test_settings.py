"""Tests for the settings & hardware-test API routes."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from robot.api.app import create_app
from robot.api.state_bridge import StateBridge
from robot.behavior.state_machine import StateMachine
from robot.config import AppSettings
from robot.events.bus import InMemoryEventBus
from robot.hardware.audio.mock_audio import MockAudioOutput
from robot.hardware.sensors.mock_camera import MockCamera
from robot.hardware.sensors.mock_microphone import MockMicrophone


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(_env_file=None, env="testing", log_level="WARNING")


@pytest.fixture
def wired_app(settings: AppSettings) -> object:
    """Create a FastAPI app with a fully-wired StateBridge (mock hardware)."""
    app = create_app(settings=settings)
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    bridge = StateBridge(
        bus=bus,
        state_machine=sm,
        camera=MockCamera(320, 240),
        microphone=MockMicrophone(sample_rate=16000, channels=1, frame_ms=30),
        audio=MockAudioOutput(sample_rate=48000, channels=1),
    )
    app.state.bridge = bridge
    return app


@pytest.fixture
async def wired_client(wired_app):
    transport = ASGITransport(app=wired_app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=10) as ac:
        yield ac


@pytest.fixture
async def client(settings: AppSettings) -> AsyncClient:  # type: ignore[misc]
    """Unwired client (no bridge components)."""
    app = create_app(settings=settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=5) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------
async def test_info_no_bridge(client: AsyncClient) -> None:
    """GET /settings/info returns ready=False when no bridge is wired."""
    r = await client.get("/api/v1/settings/info")
    assert r.status_code == 200
    d = r.json()
    assert d["ready"] is False


async def test_info_with_bridge(wired_client: AsyncClient) -> None:
    """GET /settings/info returns hardware details when bridge is wired."""
    r = await wired_client.get("/api/v1/settings/info")
    assert r.status_code == 200
    d = r.json()
    assert d["ready"] is True
    assert d["camera"]["type"] == "MockCamera"
    assert d["microphone"]["type"] == "MockMicrophone"
    assert d["audio"]["type"] == "MockAudioOutput"


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
async def test_camera_info(wired_client: AsyncClient) -> None:
    r = await wired_client.get("/api/v1/settings/camera/info")
    assert r.status_code == 200
    d = r.json()
    assert d["type"] == "MockCamera"
    assert d["width"] == 320


async def test_camera_frame(wired_client: AsyncClient) -> None:
    """GET /settings/camera/frame returns an image."""
    r = await wired_client.get("/api/v1/settings/camera/frame")
    assert r.status_code == 200
    ct = r.headers["content-type"]
    assert ct in ("image/jpeg", "image/bmp")
    assert len(r.content) > 100


async def test_camera_info_no_hardware(client: AsyncClient) -> None:
    r = await client.get("/api/v1/settings/camera/info")
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# Microphone
# ---------------------------------------------------------------------------
async def test_mic_info(wired_client: AsyncClient) -> None:
    r = await wired_client.get("/api/v1/settings/mic/info")
    assert r.status_code == 200
    d = r.json()
    assert d["type"] == "MockMicrophone"
    assert d["sample_rate"] == 16000


async def test_mic_level(wired_client: AsyncClient) -> None:
    r = await wired_client.get("/api/v1/settings/mic/level")
    assert r.status_code == 200
    d = r.json()
    assert "level" in d
    assert 0.0 <= d["level"] <= 1.0


async def test_mic_test(wired_client: AsyncClient) -> None:
    """POST /settings/mic/test records and returns WAV audio."""
    r = await wired_client.post(
        "/api/v1/settings/mic/test",
        json={"duration_s": 0.5, "play_back": False},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    # WAV files start with RIFF header.
    assert r.content[:4] == b"RIFF"
    assert len(r.content) > 1000


# ---------------------------------------------------------------------------
# Audio output
# ---------------------------------------------------------------------------
async def test_audio_info(wired_client: AsyncClient) -> None:
    r = await wired_client.get("/api/v1/settings/audio/info")
    assert r.status_code == 200
    d = r.json()
    assert d["type"] == "MockAudioOutput"
    assert d["sample_rate"] == 48000


async def test_audio_tone(wired_client: AsyncClient) -> None:
    r = await wired_client.post(
        "/api/v1/settings/audio/tone",
        json={"frequency_hz": 440, "duration_s": 0.1, "volume": 0.5},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["frequency_hz"] == 440


async def test_audio_stop(wired_client: AsyncClient) -> None:
    r = await wired_client.post("/api/v1/settings/audio/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

@pytest.mark.audio
async def test_audio_devices(wired_client: AsyncClient) -> None:
    """GET /settings/audio/devices returns a list (may be empty if no sounddevice)."""
    r = await wired_client.get("/api/v1/settings/audio/devices")
    assert r.status_code == 200

    d = r.json()
    assert "devices" in d


# ---------------------------------------------------------------------------
# Sound effects
# ---------------------------------------------------------------------------
async def test_sound_effects_list(wired_client: AsyncClient) -> None:
    r = await wired_client.get("/api/v1/settings/sound-effects")
    assert r.status_code == 200
    d = r.json()
    assert "effects" in d


# ---------------------------------------------------------------------------
# LLM / TTS (no bridge conversation - should 503)
# ---------------------------------------------------------------------------
async def test_tts_test_no_bridge(client: AsyncClient) -> None:
    r = await client.post("/api/v1/settings/tts/test", json={"text": "hello", "direct": True})
    assert r.status_code == 503


async def test_llm_test_no_bridge(client: AsyncClient) -> None:
    r = await client.post("/api/v1/settings/llm/test", json={"prompt": "hello"})
    assert r.status_code == 503
