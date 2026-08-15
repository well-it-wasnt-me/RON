"""Tests for the graceful degradation system.

Covers:
- DegradationRegistry: record, report, summary, overall_status, to_dict
- safe_init: successful init, fallback on failure, ImportError handling
- Health endpoint includes degradation info
- App starts even when every hardware component fails (all mocked)
"""

from __future__ import annotations

import time
from typing import TypedDict, cast

from robot.app import DeskBotApp
from robot.config import AppSettings, load_settings
from robot.lifecycle.degradation import (
    DegradationEntry,
    DegradationRegistry,
    safe_init,
)


class _ComponentDict(TypedDict, total=False):
    """Serialized degradation component."""

    status: str
    backend: str
    original: str
    error: str


class _DegradationDict(TypedDict):
    """Serialized degradation registry."""

    status: str
    components: dict[str, _ComponentDict]


def _typed_dict(registry: DegradationRegistry) -> _DegradationDict:
    """Return the registry serialization with its runtime shape typed."""
    return cast("_DegradationDict", registry.to_dict())


# ---------------------------------------------------------------------------
# DegradationRegistry
# ---------------------------------------------------------------------------


class TestDegradationRegistry:
    """Unit tests for DegradationRegistry."""

    def test_record_and_report(self) -> None:
        registry = DegradationRegistry()

        e1 = DegradationEntry(
            component="display",
            status="ok",
            original_backend="gc9a01",
            fallback_backend="gc9a01",
        )
        e2 = DegradationEntry(
            component="servos",
            status="degraded",
            original_backend="gpio",
            fallback_backend="mock",
            error="GPIO not available",
        )

        registry.record(e1)
        registry.record(e2)

        report = registry.report()

        assert len(report) == 2
        assert report[0].component == "display"
        assert report[0].status == "ok"
        assert report[1].component == "servos"
        assert report[1].status == "degraded"

    def test_summary_all_ok(self) -> None:
        registry = DegradationRegistry()

        registry.record(
            DegradationEntry(
                component="display",
                status="ok",
                original_backend="gc9a01",
                fallback_backend="gc9a01",
            )
        )

        assert registry.summary() == "all components ok"

    def test_summary_empty(self) -> None:
        registry = DegradationRegistry()
        assert registry.summary() == "all components ok"

    def test_summary_degraded(self) -> None:
        registry = DegradationRegistry()

        registry.record(
            DegradationEntry(
                component="servos",
                status="degraded",
                original_backend="gpio",
                fallback_backend="mock",
                error="GPIO unavailable",
            )
        )

        summary = registry.summary()

        assert "degraded" in summary
        assert "servos->mock" in summary

    def test_overall_status_ok(self) -> None:
        registry = DegradationRegistry()

        registry.record(
            DegradationEntry(
                component="display",
                status="ok",
                original_backend="gc9a01",
                fallback_backend="gc9a01",
            )
        )

        assert registry.overall_status() == "ok"

    def test_overall_status_degraded(self) -> None:
        registry = DegradationRegistry()

        registry.record(
            DegradationEntry(
                component="servos",
                status="degraded",
                original_backend="gpio",
                fallback_backend="mock",
            )
        )

        assert registry.overall_status() == "degraded"

    def test_overall_status_failed(self) -> None:
        registry = DegradationRegistry()

        registry.record(
            DegradationEntry(
                component="llm",
                status="failed",
                original_backend="openai",
                fallback_backend="mock",
            )
        )

        assert registry.overall_status() == "failed"

    def test_overall_status_empty(self) -> None:
        registry = DegradationRegistry()
        assert registry.overall_status() == "ok"

    def test_overall_status_failed_takes_priority_over_degraded(self) -> None:
        registry = DegradationRegistry()

        registry.record(
            DegradationEntry(
                component="servos",
                status="degraded",
                original_backend="gpio",
                fallback_backend="mock",
            )
        )
        registry.record(
            DegradationEntry(
                component="camera",
                status="failed",
                original_backend="usb",
                fallback_backend="mock",
            )
        )

        assert registry.overall_status() == "failed"

    def test_to_dict_ok(self) -> None:
        registry = DegradationRegistry()

        registry.record(
            DegradationEntry(
                component="display",
                status="ok",
                original_backend="gc9a01",
                fallback_backend="gc9a01",
            )
        )

        result = _typed_dict(registry)

        assert result["status"] == "ok"
        assert "display" in result["components"]
        assert result["components"]["display"]["status"] == "ok"
        assert result["components"]["display"]["backend"] == "gc9a01"

    def test_to_dict_degraded(self) -> None:
        registry = DegradationRegistry()

        registry.record(
            DegradationEntry(
                component="servos",
                status="degraded",
                original_backend="gpio",
                fallback_backend="mock",
                error="GPIO unavailable",
            )
        )

        result = _typed_dict(registry)

        assert result["status"] == "degraded"
        assert result["components"]["servos"]["backend"] == "mock"
        assert result["components"]["servos"]["original"] == "gpio"
        assert result["components"]["servos"]["error"] == "GPIO unavailable"

    def test_to_dict_empty(self) -> None:
        registry = DegradationRegistry()

        result = _typed_dict(registry)

        assert result == {
            "status": "ok",
            "components": {},
        }

    def test_report_returns_snapshot(self) -> None:
        """report() returns a copy, not a live reference."""
        registry = DegradationRegistry()

        registry.record(
            DegradationEntry(
                component="display",
                status="ok",
                original_backend="mock",
                fallback_backend="mock",
            )
        )

        report = registry.report()

        registry.record(
            DegradationEntry(
                component="servos",
                status="degraded",
                original_backend="gpio",
                fallback_backend="mock",
            )
        )

        assert len(report) == 1

    def test_entry_timestamp(self) -> None:
        """DegradationEntry defaults timestamp to time.time()."""
        before = time.time()

        entry = DegradationEntry(
            component="display",
            status="ok",
            original_backend="mock",
            fallback_backend="mock",
        )

        after = time.time()

        assert before <= entry.timestamp <= after


# ---------------------------------------------------------------------------
# safe_init
# ---------------------------------------------------------------------------


class TestSafeInit:
    """Unit tests for the safe_init helper."""

    def test_successful_init(self) -> None:
        registry = DegradationRegistry()

        result = safe_init(
            factory=lambda: "real_display",
            component="display",
            fallback=lambda: "mock_display",
            registry=registry,
            original_backend="gc9a01",
        )

        assert result == "real_display"
        assert registry.overall_status() == "ok"

    def test_fallback_on_runtime_error(self) -> None:
        registry = DegradationRegistry()

        def _fail() -> str:
            raise RuntimeError("hardware unavailable")

        result = safe_init(
            factory=_fail,
            component="servos",
            fallback=lambda: "mock_servos",
            registry=registry,
            original_backend="gpio",
            fallback_backend="mock",
        )

        assert result == "mock_servos"
        assert registry.overall_status() == "degraded"

        report = registry.report()

        assert len(report) == 1
        assert report[0].component == "servos"
        assert report[0].status == "degraded"
        assert "hardware unavailable" in (report[0].error or "")

    def test_fallback_on_import_error(self) -> None:
        registry = DegradationRegistry()

        def _fail_import() -> str:
            raise ImportError("spidev not installed")

        result = safe_init(
            factory=_fail_import,
            component="display",
            fallback=lambda: "mock_display",
            registry=registry,
            original_backend="gc9a01",
            fallback_backend="mock",
        )

        assert result == "mock_display"

        entries = registry.report()

        assert entries[0].error == "spidev not installed"

    def test_fallback_on_generic_exception(self) -> None:
        registry = DegradationRegistry()

        def _fail() -> int:
            raise ValueError("bad config")

        result = safe_init(
            factory=_fail,
            component="camera",
            fallback=lambda: 42,
            registry=registry,
            original_backend="usb",
        )

        assert result == 42
        assert registry.overall_status() == "degraded"

    def test_custom_fallback_backend_name(self) -> None:
        registry = DegradationRegistry()

        def _fail() -> str:
            raise RuntimeError("fail")

        safe_init(
            factory=_fail,
            component="wakeword",
            fallback=lambda: "null_checker",
            registry=registry,
            original_backend="porcupine",
            fallback_backend="null",
        )

        entries = registry.report()

        assert entries[0].fallback_backend == "null"

    def test_ok_entry_has_no_error(self) -> None:
        registry = DegradationRegistry()

        safe_init(
            factory=lambda: "ok",
            component="display",
            fallback=lambda: "mock",
            registry=registry,
            original_backend="gc9a01",
        )

        entries = registry.report()

        assert entries[0].error is None


# ---------------------------------------------------------------------------
# Health endpoint with degradation
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the health endpoint including degradation info."""

    def test_health_with_degradation(self) -> None:
        registry = DegradationRegistry()

        registry.record(
            DegradationEntry(
                component="servos",
                status="degraded",
                original_backend="gpio",
                fallback_backend="mock",
                error="GPIO unavailable",
            )
        )
        registry.record(
            DegradationEntry(
                component="display",
                status="ok",
                original_backend="gc9a01",
                fallback_backend="gc9a01",
            )
        )

        result = _typed_dict(registry)

        assert result["status"] == "degraded"
        assert result["components"]["display"]["status"] == "ok"
        assert result["components"]["servos"]["status"] == "degraded"

    def test_health_no_bridge(self) -> None:
        """When no bridge is available, health returns a simple ok response."""
        from robot.api.state_bridge import StateBridge

        bridge = StateBridge()

        assert bridge.degradation is None


# ---------------------------------------------------------------------------
# StateBridge integration
# ---------------------------------------------------------------------------


class TestStateBridgeDegradation:
    """Tests for DegradationRegistry on StateBridge."""

    def test_bridge_holds_degradation_registry(self) -> None:
        from robot.api.state_bridge import StateBridge

        registry = DegradationRegistry()

        registry.record(
            DegradationEntry(
                component="display",
                status="ok",
                original_backend="mock",
                fallback_backend="mock",
            )
        )

        bridge = StateBridge(degradation=registry)

        assert bridge.degradation is not None
        assert bridge.degradation.overall_status() == "ok"

    def test_bridge_default_none(self) -> None:
        from robot.api.state_bridge import StateBridge

        bridge = StateBridge()

        assert bridge.degradation is None


# ---------------------------------------------------------------------------
# App-level integration: app starts even when all hardware fails
# ---------------------------------------------------------------------------


def _mock_settings() -> AppSettings:
    """Build settings with all-mock backends and memory disabled for fast tests."""
    settings = load_settings()

    settings.hardware = "mock"
    settings.audio.backend = "mock"
    settings.tts.provider = "mock"
    settings.perception.enabled = False
    settings.learning.enabled = False
    settings.memory.enabled = False
    settings.vector_memory.enabled = False
    settings.conversation.store = "memory"
    settings.preferences.store = "memory"

    return settings


class TestAppDegradation:
    """Integration tests verifying the app uses safe_init."""

    def test_app_has_degradation_field(self) -> None:
        """DeskBotApp has a _degradation field."""
        app = DeskBotApp.build(_mock_settings())

        assert app._degradation is not None
        assert isinstance(app._degradation, DegradationRegistry)

        app._close_stores()

    def test_app_builds_with_all_mocks(self) -> None:
        """The app should build successfully with all-mock settings."""
        app = DeskBotApp.build(_mock_settings())

        assert app._degradation is not None

        report = app._degradation.report()

        component_names = {entry.component for entry in report}

        assert "display" in component_names
        assert "servos" in component_names

        app._close_stores()

    def test_app_degradation_bridge_connected(self) -> None:
        """StateBridge should share the app's DegradationRegistry."""
        app = DeskBotApp.build(_mock_settings())

        assert app._api_bridge is not None
        assert app._api_bridge.degradation is app._degradation

        app._close_stores()
