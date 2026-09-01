# Logging & observability

DeskBot uses [structlog](https://www.structlog.org/) for structured JSON
logging, plus an in-memory **ring buffer** that feeds the web dashboard's
`/#/logs` view and the dashboard "Recent Events" feed.

## Getting a logger

Application code never constructs loggers directly — it calls
`robot.logging.get_logger(name)`, which prefixes the name with `robot.` and
binds it into the event context:

```python
from robot.logging import get_logger

_log = get_logger("behavior")
_log.info("state_changed", previous="idle", current="curious")
```

This renders a single JSON line to stdout:

```json
{"previous": "idle", "current": "curious", "event": "state_changed", "level": "info", "timestamp": "2026-08-31T22:10:00Z"}
```

`configure_logging()` is called once at startup and is idempotent. It wires
structlog (`PrintLoggerFactory` + `JSONRenderer`) and the stdlib root logger.

## The ring buffer — and why it needed a processor

The dashboard's log table is backed by `_RingBufferHandler`, an in-memory
ring buffer of recent `LogEntry` objects. The subtle part: structlog's
`PrintLoggerFactory` writes JSON **straight to stdout, bypassing the stdlib
logging tree**. A plain `logging.Handler` attached to the stdlib root logger
therefore only catches third-party stdlib logs (uvicorn, httpx, ...) and
misses almost every DeskBot event — which is exactly why `/#/logs` used to
show only a couple of stale records.

The fix is a structlog processor, `_capture_to_ring_buffer`, inserted into
the processor chain **before** `JSONRenderer`. It builds a `LogEntry` directly
from the event dict and appends it to the ring buffer, so every structured
event reaches the buffer regardless of the logger factory. The stdlib
`_RingBufferHandler` is kept too, so third-party stdlib logs are also
captured. Both paths feed the same `deque`.

The logger name is carried through via `get_logger(...).bind(logger_name=...)`
and stripped by a `_strip_logger_name` processor before rendering, so the
stdout JSON is unchanged while the buffer records the source module.

## `LogEntry`

Each captured entry has:

| Field | Type | Description |
|---|---|---|
| `timestamp` | `str` | ISO-8601 UTC (`...Z`) |
| `created_epoch` | `float` | POSIX epoch, for `since`/live-tail filtering |
| `level` | `str` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `logger_name` | `str` | e.g. `robot.behavior` |
| `event` | `str` | The event/message name |
| `data` | `dict` | Remaining structured payload |

## Filtering

### Logs API (`GET /api/v1/system/logs`)

`_RingBufferHandler.get_entries()` filters server-side:

- `level` — `DEBUG`/`INFO`/`WARNING`/`ERROR` (or `ALL` / empty for no filter)
- `search` — case-insensitive substring across event, logger name, and data values
- `logger` — case-insensitive substring on the logger name
- `event` — case-insensitive substring on the event name
- `exclude` — event-name deny list (case-insensitive exact match)
- `since_epoch` — only entries with `created_epoch >=` this (live tailing)
- `limit` — most-recent N (default 200, max 500)

`GET /api/v1/system/logs/filters` returns the distinct `levels`, `loggers`,
and `events` currently buffered, for populating dashboard dropdowns.
`DELETE /api/v1/system/logs` clears the buffer.

### WebSocket event stream

The dashboard "Recent Events" feed is driven by the WebSocket event stream,
which supports **per-connection filtering** (see
[Events](events.md#websocket-streaming-filtering) and
[REST API &middot; WebSocket](../reference/api.md#websocket)). The dashboard
hides the high-frequency "noisy" events by default and reconnects with an
`?exclude=` query param when the toggle changes.

## Configuration (`LoggingConfig`)

Ring-buffer capacity and the default "noisy events" hide list are
env-tunable via the `logging` block on `AppSettings`
(`DESKBOT_LOGGING__` prefix):

```env
# In-memory ring-buffer size (FIFO; 10-10000):
DESKBOT_LOGGING__RING_BUFFER_CAPACITY=500
# Event type names hidden by default in the dashboard feed (comma-separated):
DESKBOT_LOGGING__NOISY_EVENTS=DisplayUpdated,LookRequested,BlinkRequested,ServoMoved,IdleTimeout,LookAroundAction,FaceDetected
```

See also [Configuration](../reference/config.md#logging) and the
[REST API reference](../reference/api.md#system).