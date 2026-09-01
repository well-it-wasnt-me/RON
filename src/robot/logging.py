"""Structured logging for DeskBot.

Every subsystem uses its own logger (``robot.eye_engine``, ``robot.behavior``,
...) and a single :func:`configure_logging` call sets up handlers, formatting,
and ISO-8601 timestamps. :func:`get_logger` is the only public entry point
application code should use.

In addition to rendering JSON to stdout, every structured event is captured
into an in-memory **ring buffer** (:class:`_RingBufferHandler`) that feeds
the web dashboard's ``/#/logs`` view. The capture happens via a structlog
processor (:func:`_capture_to_ring_buffer`) so events reach the buffer
regardless of the configured logger factory — structlog's
``PrintLoggerFactory`` writes straight to stdout and bypasses the stdlib
logging tree, so a plain ``logging.Handler`` on the root logger alone would
miss almost every DeskBot log. The stdlib handler is kept too, so
third-party stdlib loggers (uvicorn, httpx, ...) are also captured.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, TextIO, cast

import structlog
from structlog.types import EventDict, WrappedLogger

from robot.config import AppSettings

DEFAULT_LEVEL: Final[int] = logging.INFO
#: Default ring-buffer capacity when no :class:`AppSettings` is provided.
DEFAULT_RING_BUFFER_CAPACITY: Final[int] = 500

#: structlog event-dict keys that are structural metadata, not payload.
_META_KEYS: Final[frozenset[str]] = frozenset(
    {"event", "level", "logger_name", "timestamp"}
)


class _LogState:
    """Holds the configuration flag (avoids module-level ``global``)."""

    configured: bool = False


def _iso_now() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_from_epoch(epoch: float) -> str:
    """Return *epoch* (seconds) as an ISO-8601 UTC string with a ``Z`` suffix."""
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch_from_iso(timestamp: str) -> float:
    """Parse an ISO-8601 timestamp into a POSIX epoch float.

    Falls back to ``time.time()`` if the string cannot be parsed so the
    ring buffer never drops an entry due to a malformed timestamp.
    """
    try:
        text = timestamp.replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    except (ValueError, TypeError):
        return datetime.now(tz=UTC).timestamp()


def configure_logging(
    settings: AppSettings | None = None,
    *,
    stream: TextIO | None = None,
    force: bool = False,
) -> None:
    """Configure structlog + stdlib logging exactly once.

    Parameters
    ----------
    settings:
        Optional :class:`~robot.config.AppSettings`. When ``None`` the default
        ``INFO`` level and ring-buffer capacity are used.
    stream:
        Optional output stream for the JSON renderer (defaults to stdout).
    force:
        When ``True``, reconfigure even if logging was already configured.
        Intended for tests that need a fresh ring buffer.
    """
    if _LogState.configured and not force:
        return

    _LogState.configured = False

    log_stream = stream if stream is not None else sys.stdout

    level_name = settings.log_level if settings is not None else "INFO"
    level = getattr(logging, level_name.upper(), DEFAULT_LEVEL)

    capacity = DEFAULT_RING_BUFFER_CAPACITY
    if settings is not None:
        capacity = int(getattr(settings.logging, "ring_buffer_capacity", capacity))

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
            _capture_to_ring_buffer,
            _strip_logger_name,
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
    install_ring_buffer(capacity)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for the given module name.

    The name is prefixed with ``robot.`` if it doesn't already start with it,
    so log records naturally form a hierarchy. The name is also bound into
    the event context as ``logger_name`` so :func:`_capture_to_ring_buffer`
    can record it; :func:`_strip_logger_name` removes it before rendering so
    the stdout JSON output stays unchanged.
    """
    if not name.startswith("robot."):
        name = f"robot.{name}"
    logger = structlog.get_logger(name).bind(logger_name=name)
    return cast("structlog.stdlib.BoundLogger", logger)


__all__ = ["configure_logging", "get_logger"]


# ---------------------------------------------------------------------------
# structlog processor: capture every event into the ring buffer
# ---------------------------------------------------------------------------


def _capture_to_ring_buffer(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Structlog processor that appends each event to the dashboard ring buffer.

    Runs before :func:`_strip_logger_name` and ``JSONRenderer`` so the
    ``logger_name`` and full payload are still available. Any failure is
    swallowed — logging must never crash the application.
    """
    try:
        raw_level = str(event_dict.get("level", method_name))
        level = raw_level.upper() if raw_level == raw_level.lower() else raw_level
        event = str(event_dict.get("event", ""))
        logger_name = str(event_dict.get("logger_name", "robot"))
        ts = event_dict.get("timestamp")
        timestamp = str(ts) if ts else _iso_now()
        data = {k: v for k, v in event_dict.items() if k not in _META_KEYS}
        get_ring_buffer().add_entry(
            LogEntry(
                timestamp=timestamp,
                created_epoch=_epoch_from_iso(timestamp),
                level=level or method_name.upper(),
                logger_name=logger_name,
                event=event,
                data=data,
            )
        )
    except Exception:
        pass
    return event_dict


def _strip_logger_name(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Remove the internal ``logger_name`` key before rendering to stdout."""
    event_dict.pop("logger_name", None)
    return event_dict


# ---------------------------------------------------------------------------
# In-memory log ring buffer for the web dashboard
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LogEntry:
    """A single captured log entry for the dashboard."""

    timestamp: str
    created_epoch: float
    level: str
    logger_name: str
    event: str
    data: dict[str, Any]


class _RingBufferHandler(logging.Handler):
    """A logging handler that stores recent entries in a ring buffer.

    Captures structured log events for the web dashboard's live log view.
    Structlog events are fed in via :func:`_capture_to_ring_buffer`; stdlib
    log records (third-party libraries) are captured through :meth:`emit`.
    """

    def __init__(self, capacity: int = DEFAULT_RING_BUFFER_CAPACITY) -> None:
        super().__init__()
        self._entries: deque[LogEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    # -- structlog path --------------------------------------------------
    def add_entry(self, entry: LogEntry) -> None:
        """Append a pre-built entry (called by the structlog processor)."""
        with self._lock:
            self._entries.append(entry)

    # -- stdlib path -----------------------------------------------------
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            event = msg
            data: dict[str, Any] = {}

            # structlog may have logged through stdlib as a JSON string; try
            # to recover the structured payload. Otherwise use the plain
            # formatted message.
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

            self.add_entry(
                LogEntry(
                    timestamp=_iso_from_epoch(record.created),
                    created_epoch=float(record.created),
                    level=record.levelname,
                    logger_name=record.name,
                    event=event,
                    data=data,
                )
            )
        except Exception:
            pass

    # -- querying --------------------------------------------------------
    def get_entries(
        self,
        level: str | None = None,
        search: str | None = None,
        limit: int = 200,
        logger: str | None = None,
        event: str | None = None,
        exclude: list[str] | None = None,
        since_epoch: float | None = None,
    ) -> list[LogEntry]:
        """Return recent log entries, optionally filtered.

        Parameters
        ----------
        level:
            Filter by level (``DEBUG``/``INFO``/``WARNING``/``ERROR``).
            ``"ALL"`` or empty means no level filter.
        search:
            Case-insensitive substring matched against the event, logger
            name, and data values.
        logger:
            Case-insensitive substring matched against the logger name.
        event:
            Case-insensitive substring matched against the event name.
        exclude:
            Event names to omit (case-insensitive exact match). Used by the
            dashboard to hide noisy events.
        since_epoch:
            Only entries with ``created_epoch`` >= this POSIX timestamp
            (used for live-tail polling).
        limit:
            Maximum number of entries to return (most recent first).
        """
        with self._lock:
            entries = list(self._entries)

        exclude_set = {name.lower() for name in exclude} if exclude else set()

        if level and level.upper() not in ("ALL", ""):
            wanted = level.upper()
            entries = [e for e in entries if e.level == wanted]
        if since_epoch is not None:
            entries = [e for e in entries if e.created_epoch >= since_epoch]
        if exclude_set:
            entries = [e for e in entries if e.event.lower() not in exclude_set]
        if logger:
            needle = logger.lower()
            entries = [e for e in entries if needle in e.logger_name.lower()]
        if event:
            needle = event.lower()
            entries = [e for e in entries if needle in e.event.lower()]
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

    def distinct_filters(self) -> dict[str, list[str]]:
        """Return the distinct levels, logger names, and event names present.

        Used by the dashboard to populate filter dropdowns.
        """
        with self._lock:
            entries = list(self._entries)
        return {
            "levels": sorted({e.level for e in entries if e.level}),
            "loggers": sorted({e.logger_name for e in entries if e.logger_name}),
            "events": sorted({e.event for e in entries if e.event}),
        }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# Module-level singleton.
_ring_buffer: _RingBufferHandler | None = None


def get_ring_buffer() -> _RingBufferHandler:
    """Return the module-level ring buffer handler, creating it if needed."""
    global _ring_buffer  # noqa: PLW0603
    if _ring_buffer is None:
        _ring_buffer = _RingBufferHandler(capacity=DEFAULT_RING_BUFFER_CAPACITY)
    return _ring_buffer


def install_ring_buffer(capacity: int = DEFAULT_RING_BUFFER_CAPACITY) -> _RingBufferHandler:
    """Create and attach the ring buffer handler to the root logger.

    Call after :func:`configure_logging` to start capturing log entries
    for the web dashboard. Replaces any previously-installed ring buffer so
    ``configure_logging(force=True)`` gets a fresh buffer.
    """
    global _ring_buffer  # noqa: PLW0603
    handler = _RingBufferHandler(capacity=capacity)
    _ring_buffer = handler
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, _RingBufferHandler)]
    root.addHandler(handler)
    return handler
