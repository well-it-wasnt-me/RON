"""Tests for config validation API endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from robot.api.app import create_app
from robot.config import AppSettings


@pytest.fixture
def settings() -> AppSettings:
    """Create test settings."""
    return AppSettings(
        _env_file=None,
        env="testing",
        log_level="WARNING",
    )


@pytest.fixture
def app(settings: AppSettings) -> FastAPI:
    """Create a test FastAPI app."""
    return create_app(settings=settings)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as ac:
        yield ac


# ------------------------------------------------------- config/validate POST


async def test_validate_valid_config(client: AsyncClient) -> None:
    """POST /api/v1/config/validate with valid config returns valid=true."""
    response = await client.post(
        "/api/v1/config/validate",
        json={"env": "testing"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["valid"] is True


async def test_validate_empty_config(client: AsyncClient) -> None:
    """POST /api/v1/config/validate with empty body returns valid=true."""
    response = await client.post(
        "/api/v1/config/validate",
        json={},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["valid"] is True


async def test_validate_invalid_env(client: AsyncClient) -> None:
    """POST /api/v1/config/validate with invalid env returns field-level errors."""
    response = await client.post(
        "/api/v1/config/validate",
        json={"env": "invalid_env"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["valid"] is False
    assert len(data["errors"]) > 0

    env_errors = [
        error
        for error in data["errors"]
        if "env" in error["field"]
    ]
    assert len(env_errors) > 0


async def test_validate_invalid_port(client: AsyncClient) -> None:
    """POST /api/v1/config/validate with out-of-range port returns errors."""
    response = await client.post(
        "/api/v1/config/validate",
        json={"api": {"port": 99999}},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["valid"] is False

    port_errors = [
        error
        for error in data["errors"]
        if "port" in error["field"]
    ]
    assert len(port_errors) > 0


async def test_validate_invalid_spi_hz(client: AsyncClient) -> None:
    """POST /api/v1/config/validate with too-low spi_hz returns errors."""
    response = await client.post(
        "/api/v1/config/validate",
        json={"displays": {"spi_hz": 50}},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["valid"] is False

    hz_errors = [
        error
        for error in data["errors"]
        if "spi_hz" in error["field"]
    ]
    assert len(hz_errors) > 0


async def test_validate_partial_override(client: AsyncClient) -> None:
    """POST /api/v1/config/validate with a partial nested override works."""
    response = await client.post(
        "/api/v1/config/validate",
        json={"llm": {"temperature": 0.3}},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["valid"] is True


async def test_validate_does_not_modify_running_config(
    client: AsyncClient,
) -> None:
    """POST /api/v1/config/validate must not modify the running config."""
    before = await client.get("/api/v1/config")

    assert before.status_code == 200

    config_before = before.json()

    response = await client.post(
        "/api/v1/config/validate",
        json={"env": "production"},
    )

    assert response.status_code == 200
    assert response.json()["valid"] is True

    after = await client.get("/api/v1/config")

    assert after.status_code == 200

    config_after = after.json()

    assert config_after["env"] == config_before["env"]


async def test_validate_multiple_errors(client: AsyncClient) -> None:
    """POST /api/v1/config/validate returns all validation errors."""
    response = await client.post(
        "/api/v1/config/validate",
        json={
            "env": "invalid_env",
            "api": {"port": 99999},
            "personality": {"curiosity": 5.0},
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["valid"] is False
    assert len(data["errors"]) >= 3


async def test_validate_api_key_not_applied(
    client: AsyncClient,
) -> None:
    """Validation must not leak an API key into running config."""
    before = await client.get("/api/v1/config")

    assert before.status_code == 200

    response = await client.post(
        "/api/v1/config/validate",
        json={
            "llm": {
                "api_key": "sk-test-key-1234567890",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["valid"] is True

    after = await client.get("/api/v1/config")

    assert after.status_code == 200

    assert "sk-test-key" not in (
        after.json()
        .get("llm", {})
        .get("api_key", "")
    )


# ------------------------------------------------------------ config/schema GET


async def test_get_config_schema(client: AsyncClient) -> None:
    """GET /api/v1/config/schema returns the full Pydantic schema."""
    response = await client.get("/api/v1/config/schema")

    assert response.status_code == 200

    data = response.json()
    assert "properties" in data


async def test_schema_contains_main_sections(
    client: AsyncClient,
) -> None:
    """The schema should contain major config sections."""
    response = await client.get("/api/v1/config/schema")

    assert response.status_code == 200

    schema = response.json()
    props = schema.get("properties", {})

    assert "env" in props


async def test_schema_error_format() -> None:
    """Validation errors contain field, message, type, and input."""
    settings = AppSettings(
        _env_file=None,
        env="testing",
        log_level="WARNING",
    )

    app = create_app(settings=settings)

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as ac:
        response = await ac.post(
            "/api/v1/config/validate",
            json={"env": "not_a_valid_env"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["valid"] is False

    errors = data["errors"]
    assert len(errors) > 0

    error = errors[0]

    assert "field" in error
    assert "message" in error
    assert "type" in error
    assert "input" in error
