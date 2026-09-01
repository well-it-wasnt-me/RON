"""Tests for the ``/api/v1/system/logs`` endpoints.

Covers the new filter params, the ``/logs/filters`` dropdown endpoint, and
the clear endpoint. The ring buffer is populated directly so the tests are
deterministic and independent of the structlog configuration.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from robot.api.app import create_app
from robot.config import AppSettings
from robot.logging import LogEntry, get_ring_buffer


def _entry(
    *,
    event: str,
    level: str = "INFO",
    logger_name: str = "robot.test",
    epoch: float = 1000.0,
    data: dict[str, Any] | None = None,
) -> LogEntry:
    return LogEntry(
        timestamp=f"t-{epoch}",
        created_epoch=epoch,
        level=level,
        logger_name=logger_name,
        event=event,
        data=data or {},
    )


def _populate_buffer() -> None:
    rb = get_ring_buffer()
    rb.clear()
    rb.add_entry(_entry(event="DisplayUpdated", level="DEBUG", logger_name="robot.face", epoch=1.0))
    rb.add_entry(_entry(event="StateChanged", level="INFO", logger_name="robot.behavior", epoch=2.0))
    rb.add_entry(_entry(event="LookRequested", level="INFO", logger_name="robot.behavior", epoch=3.0))
    rb.add_entry(
        _entry(event="RobotError", level="ERROR", logger_name="robot.core", epoch=4.0, data={"reason": "boom"})
    )


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(_env_file=None, env="testing", log_level="WARNING")


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_logs_return_all(settings: AppSettings) -> None:
    _populate_buffer()
    app = create_app(settings=settings)
    try:
        async with _client(app) as c:
            r = await c.get("/api/v1/system/logs?limit=100")
            assert r.status_code == 200
            data = r.json()
            assert data["count"] == 4
            assert data["entries"][0]["event"] == "DisplayUpdated"
            # created_epoch is present on each entry.
            assert "created_epoch" in data["entries"][0]
    finally:
        get_ring_buffer().clear()


async def test_logs_filter_by_level(settings: AppSettings) -> None:
    _populate_buffer()
    app = create_app(settings=settings)
    try:
        async with _client(app) as c:
            r = await c.get("/api/v1/system/logs?level=ERROR")
            assert r.status_code == 200
            data = r.json()
            assert data["count"] == 1
            assert data["entries"][0]["event"] == "RobotError"
    finally:
        get_ring_buffer().clear()


async def test_logs_filter_by_logger(settings: AppSettings) -> None:
    _populate_buffer()
    app = create_app(settings=settings)
    try:
        async with _client(app) as c:
            r = await c.get("/api/v1/system/logs?logger=behavior")
            assert {e["event"] for e in r.json()["entries"]} == {"StateChanged", "LookRequested"}
    finally:
        get_ring_buffer().clear()


async def test_logs_filter_by_event(settings: AppSettings) -> None:
    _populate_buffer()
    app = create_app(settings=settings)
    try:
        async with _client(app) as c:
            r = await c.get("/api/v1/system/logs?event=state")
            assert [e["event"] for e in r.json()["entries"]] == ["StateChanged"]
    finally:
        get_ring_buffer().clear()


async def test_logs_exclude_deny_list(settings: AppSettings) -> None:
    _populate_buffer()
    app = create_app(settings=settings)
    try:
        async with _client(app) as c:
            r = await c.get("/api/v1/system/logs?exclude=DisplayUpdated,LookRequested&limit=100")
            events = {e["event"] for e in r.json()["entries"]}
            assert events == {"StateChanged", "RobotError"}
    finally:
        get_ring_buffer().clear()


async def test_logs_since_epoch(settings: AppSettings) -> None:
    _populate_buffer()
    app = create_app(settings=settings)
    try:
        async with _client(app) as c:
            r = await c.get("/api/v1/system/logs?since=3.0&limit=100")
            events = {e["event"] for e in r.json()["entries"]}
            assert events == {"LookRequested", "RobotError"}
    finally:
        get_ring_buffer().clear()


async def test_logs_filters_endpoint(settings: AppSettings) -> None:
    _populate_buffer()
    app = create_app(settings=settings)
    try:
        async with _client(app) as c:
            r = await c.get("/api/v1/system/logs/filters")
            assert r.status_code == 200
            data = r.json()
            assert data["levels"] == ["DEBUG", "ERROR", "INFO"]
            assert "robot.behavior" in data["loggers"]
            assert "StateChanged" in data["events"]
            # The server-configured noisy-events hide list is echoed back so
            # the dashboard can mirror it without hardcoding the names in JS.
            assert "FaceDetected" in data["noisy_events"]
            assert "DisplayUpdated" in data["noisy_events"]
    finally:
        get_ring_buffer().clear()


async def test_logs_delete_clears_buffer(settings: AppSettings) -> None:
    _populate_buffer()
    app = create_app(settings=settings)
    try:
        async with _client(app) as c:
            # api_key is empty in these settings, so no auth required.
            r = await c.delete("/api/v1/system/logs")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"
            # Buffer is now empty within the same app instance.
            r2 = await c.get("/api/v1/system/logs?limit=100")
            assert r2.json()["count"] == 0
    finally:
        get_ring_buffer().clear()
