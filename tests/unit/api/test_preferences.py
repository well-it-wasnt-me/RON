"""Tests for the preferences REST API endpoints."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from robot.ai.preferences import InMemoryPreferenceStore, PreferenceTracker
from robot.api.app import create_app
from robot.api.state_bridge import StateBridge
from robot.behavior.state_machine import StateMachine
from robot.config import AppSettings
from robot.events.bus import InMemoryEventBus


def _make_app_with_tracker() -> object:
    """Create a FastAPI app with a wired preference tracker."""
    settings = AppSettings(_env_file=None, env="testing", log_level="WARNING")
    app = create_app(settings=settings)
    bus = InMemoryEventBus()
    state_machine = StateMachine(bus=bus)
    tracker = PreferenceTracker(store=InMemoryPreferenceStore())
    tracker.process_user_text("My name is Alice")
    bridge = StateBridge(
        bus=bus,
        state_machine=state_machine,
        preference_tracker=tracker,
    )
    app.state.bridge = bridge
    return app


@pytest.fixture
def app_with_tracker() -> object:
    return _make_app_with_tracker()


@pytest.fixture
async def client(app_with_tracker: object) -> AsyncClient:  # type: ignore[misc]
    transport = ASGITransport(app=app_with_tracker)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestPreferencesAPI:
    @pytest.mark.asyncio
    async def test_list_preferences(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/preferences")
        assert response.status_code == 200
        data = response.json()
        assert "preferences" in data
        assert len(data["preferences"]) >= 1
        # Should contain the "name" preference
        names = [p["key"] for p in data["preferences"]]
        assert "name" in names

    @pytest.mark.asyncio
    async def test_get_preference(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/preferences/name")
        assert response.status_code == 200
        data = response.json()
        assert data["key"] == "name"
        assert data["value"] == "alice"
        assert data["source"] == "explicit"

    @pytest.mark.asyncio
    async def test_get_missing_preference(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/preferences/missing_key")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_preference(self, client: AsyncClient) -> None:
        response = await client.delete("/api/v1/preferences/name")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["key"] == "name"

        # Verify it's gone.
        response = await client.get("/api/v1/preferences/name")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_missing_preference(self, client: AsyncClient) -> None:
        response = await client.delete("/api/v1/preferences/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_preferences_not_available_without_tracker(self) -> None:
        """When preference_tracker is None, the API returns 404."""
        settings = AppSettings(_env_file=None, env="testing", log_level="WARNING")
        app = create_app(settings=settings)
        bus = InMemoryEventBus()
        state_machine = StateMachine(bus=bus)
        bridge = StateBridge(bus=bus, state_machine=state_machine, preference_tracker=None)
        app.state.bridge = bridge
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/api/v1/preferences")
            assert response.status_code == 404
