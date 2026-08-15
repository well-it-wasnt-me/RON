"""Structured logging for DeskBot.

Every subsystem uses its own logger (``robot.eye_engine``, ``robot.behavior``,
...) and a single :func:`configure_logging` call sets up handlers, formatting,
and ISO-8601 timestamps. :func:`get_logger` is the only public entry point
application code should use.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Final, TextIO, cast

import structlog

from robot.config import AppSettings

DEFAULT_LEVEL: Final[int] = logging.INFO


class _LogState:
    """Holds the configuration flag (avoids module-level ``global``)."""

    configured: bool = False


def configure_logging(
    settings: AppSettings | None = None,
    *,
    stream: TextIO | None = None,
) -> None:
    """Configure structlog + stdlib logging exactly once.

    Parameters
    ----------
    settings:
        Optional :class:`~robot.config.AppSettings`. When ``None`` the default
        ``INFO`` level is used.
    """
    if _LogState.configured:
        return

    log_stream = stream if stream is not None else sys.stdout

    level_name = settings.log_level if settings is not None else "INFO"
    level = getattr(logging, level_name.upper(), DEFAULT_LEVEL)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=log_stream),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=log_stream,
        force=True,
    )

    # Quiet down noisy 3rd-party loggers.
    for noisy in ("uvicorn", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _LogState.configured = True

    # Attach the ring buffer handler for the web dashboard.
    install_ring_buffer()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for the given module name.

    The name is prefixed with ``robot.`` if it doesn't already start with it,
    so log records naturally form a hierarchy.
    """
    if not name.startswith("robot."):
        name = f"robot.{name}"
    logger = structlog.get_logger(name)
    # The first call returns a lazy proxy; force resolution by calling .bind()
    # once so the returned object is the real BoundLogger.
    return cast("structlog.stdlib.BoundLogger", logger)


__all__ = ["configure_logging", "get_logger"]


# ---------------------------------------------------------------------------
# In-memory log ring buffer for the web dashboard
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LogEntry:
    """A single captured log entry for the dashboard."""

    timestamp: str
    level: str
    logger_name: str
    event: str
    data: dict[str, Any]


class _RingBufferHandler(logging.Handler):
    """A logging handler that stores recent entries in a ring buffer.

    Captures structured log events for the web dashboard's live log view.
    """

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self._entries: deque[LogEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # structlog renders to JSON on stdout, but the LogRecord
            # still carries the original event dict in `record.msg` when
            # using structlog's PrintLoggerFactory.  For stdlib loggers,
            # record.getMessage() gives the formatted message.
            msg = record.getMessage()
            event = msg
            data: dict[str, Any] = {}

            # structlog stores the event dict as `record.msg` (a dict)
            # before rendering.  When using JSONRenderer, the rendered
            # string is what ends up in `record.msg`.  We try to parse
            # it back; if that fails, we use the plain message.
            if isinstance(record.msg, dict):
                event = str(record.msg.get("event", record.getMessage()))
                data = {k: v for k, v in record.msg.items() if k != "event"}
            elif isinstance(record.msg, str):
                import json

                try:
                    parsed = json.loads(record.msg)
                    if isinstance(parsed, dict):
                        event = str(parsed.get("event", record.msg))
                        data = {k: v for k, v in parsed.items() if k != "event"}
                except (json.JSONDecodeError, TypeError):
                    event = record.msg

            entry = LogEntry(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
                level=record.levelname,
                logger_name=record.name,
                event=event,
                data=data,
            )
            with self._lock:
                self._entries.append(entry)
        except Exception:
            pass  # never let logging crash the app

    def get_entries(
        self,
        level: str | None = None,
        search: str | None = None,
        limit: int = 200,
    ) -> list[LogEntry]:
        """Return recent log entries, optionally filtered."""
        with self._lock:
            entries = list(self._entries)
        if level and level.upper() not in ("ALL", ""):
            entries = [e for e in entries if e.level == level.upper()]
        if search:
            search_lower = search.lower()
            entries = [
                e
                for e in entries
                if search_lower in e.event.lower()
                or search_lower in e.logger_name.lower()
                or any(search_lower in str(v).lower() for v in e.data.values())
            ]
        return entries[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# Module-level singleton.
_ring_buffer: _RingBufferHandler | None = None


def get_ring_buffer() -> _RingBufferHandler:
    """Return the module-level ring buffer handler, creating it if needed."""
    global _ring_buffer  # noqa: PLW0603
    if _ring_buffer is None:
        _ring_buffer = _RingBufferHandler(capacity=500)
    return _ring_buffer


def install_ring_buffer() -> _RingBufferHandler:
    """Create and attach the ring buffer handler to the root logger.

    Call after :func:`configure_logging` to start capturing log entries
    for the web dashboard.
    """
    handler = get_ring_buffer()
    root = logging.getLogger()
    if handler not in root.handlers:
        root.addHandler(handler)
    return handler
