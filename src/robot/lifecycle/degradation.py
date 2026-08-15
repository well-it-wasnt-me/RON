"""Graceful degradation registry for hardware component failures.

When a hardware component (display, servos, microphone, etc.) fails to
initialise, the application falls back to a mock implementation and records
the failure in the :class:`DegradationRegistry`.  The registry is accessible
via the ``GET /api/v1/health`` endpoint so that external monitoring tools
can see which components are running in degraded mode.

Design principles
-----------------
* The robot must **never crash** due to a hardware failure.
* Every component has a mock fallback.
* Degradation entries are logged at ``WARNING`` level, not ``ERROR``.
* The registry is accessible from the API even in ``ERROR`` state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from robot.logging import get_logger

_log = get_logger("lifecycle.degradation")

Status = Literal["ok", "degraded", "failed"]

T = TypeVar("T")


@dataclass(slots=True)
class DegradationEntry:
    """One recorded component degradation event."""

    component: str
    status: Status
    original_backend: str
    fallback_backend: str
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


class DegradationRegistry:
    """Tracks component degradation across the application.

    Thread-safe for reads: :meth:`report` returns a snapshot that will
    not mutate under the caller.  Concurrent :meth:`record` calls are
    safe because Python's GIL serialises list appends.
    """

    def __init__(self) -> None:
        self._entries: list[DegradationEntry] = []

    def record(self, entry: DegradationEntry) -> None:
        """Record a degradation entry and log at WARNING level."""
        self._entries.append(entry)
        if entry.status == "ok":
            _log.info(
                "component.ok",
                component=entry.component,
                backend=entry.original_backend,
            )
        else:
            _log.warning(
                "component.fallback",
                component=entry.component,
                original=entry.original_backend,
                fallback=entry.fallback_backend,
                error=entry.error,
            )

    def report(self) -> list[DegradationEntry]:
        """Return a snapshot of all recorded degradation entries."""
        return list(self._entries)

    def summary(self) -> str:
        """Return a human-readable one-line summary of degradation status."""
        if not self._entries:
            return "all components ok"
        degraded = [e for e in self._entries if e.status != "ok"]
        if not degraded:
            return "all components ok"
        parts = [f"{e.component}->{e.fallback_backend}" for e in degraded]
        return "degraded: " + ", ".join(parts)

    def overall_status(self) -> Status:
        """Return the worst status across all entries.

        Returns ``"ok"`` if no entries have been recorded yet.
        """
        if not self._entries:
            return "ok"
        if any(e.status == "failed" for e in self._entries):
            return "failed"
        if any(e.status == "degraded" for e in self._entries):
            return "degraded"
        return "ok"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict for the health endpoint."""
        components: dict[str, dict[str, str]] = {}
        for entry in self._entries:
            info: dict[str, str] = {
                "status": entry.status,
                "backend": (
                    entry.original_backend if entry.status == "ok" else entry.fallback_backend
                ),
            }
            if entry.original_backend != entry.fallback_backend:
                info["original"] = entry.original_backend
            if entry.error is not None:
                info["error"] = entry.error
            components[entry.component] = info
        return {
            "status": self.overall_status(),
            "components": components,
        }


def safe_init[T](
    factory: Callable[[], T],
    component: str,
    fallback: Callable[[], T],
    registry: DegradationRegistry,
    *,
    original_backend: str,
    fallback_backend: str = "mock",
) -> T:
    """Try to create a component; fall back to *fallback* on failure.

    Parameters
    ----------
    factory:
        Callable that returns the real component.  May raise
        :class:`ImportError` or :class:`RuntimeError` (or any
        exception) if the hardware is unavailable.
    component:
        Human-readable component name (e.g. ``"display"``).
    fallback:
        Callable that returns the mock fallback instance.
    registry:
        :class:`DegradationRegistry` to record the outcome.
    original_backend:
        Name of the backend that was attempted (e.g. ``"gc9a01"``).
    fallback_backend:
        Name of the fallback backend (defaults to ``"mock"``).

    Returns
    -------
    The real instance if *factory* succeeds, otherwise the mock.
    """
    try:
        instance = factory()
        registry.record(
            DegradationEntry(
                component=component,
                status="ok",
                original_backend=original_backend,
                fallback_backend=original_backend,
            )
        )
        return instance
    except Exception as exc:
        _log.warning(
            "component.init_fallback",
            component=component,
            original=original_backend,
            fallback=fallback_backend,
            error=str(exc),
        )
        registry.record(
            DegradationEntry(
                component=component,
                status="degraded",
                original_backend=original_backend,
                fallback_backend=fallback_backend,
                error=str(exc),
            )
        )
        return fallback()


__all__ = [
    "DegradationEntry",
    "DegradationRegistry",
    "Status",
    "safe_init",
]
