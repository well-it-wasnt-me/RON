# Performance profiling

DeskBot includes lightweight performance profiling that lets you measure and
monitor frame budget compliance, servo latency, event bus throughput, and
identify bottlenecks.

> See also the [Performance reference](../reference/performance.md) for the
> detailed profiler stats, Pi 5 baselines, and overhead analysis.

---

## Components

| Profiler | What it measures | Location |
|----------|-----------------|----------|
| `FrameProfiler` | Frame rendering time vs. budget | `robot.performance.frame_profiler` |
| `ServoProfiler` | Servo command latency | `robot.performance.servo_profiler` |
| `BusProfiler` | Event bus throughput & processing time | `robot.performance.bus_profiler` |

Each profiler uses rolling windows (last N entries) so memory usage is constant
regardless of runtime. All timing uses `time.monotonic()`.

---

## Configuration

Performance profiling is controlled via `PerformanceConfig` (environment
prefix: `DESKBOT_PERFORMANCE__`):

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `True` | Master switch. When `False`, all profilers are no-ops with zero overhead. |
| `frame_profiling` | `True` | Enable frame budget monitoring. |
| `servo_profiling` | `True` | Enable servo latency profiling. |
| `bus_profiling` | `True` | Enable event bus throughput profiling. |
| `bus_sample_rate` | `0.1` | Fraction of bus events to sample (10% = 1 in 10). |
| `report_interval_frames` | `100` | Publish `FrameStatsReport` event every N frames. |

```bash
# Disable all profiling
DESKBOT_PERFORMANCE__ENABLED=false

# Sample 25% of bus events
DESKBOT_PERFORMANCE__BUS_SAMPLE_RATE=0.25
```

---

## Integration

Profilers are wired into `DeskBotApp` during `build()`. When enabled, they
attach to the event bus and publish periodic `FrameStatsReport` events. The
REST API exposes all profiling data at `/api/v1/performance/*`.

When disabled, each profiler's `record_*()` method returns immediately (single
boolean check) — zero overhead.

---

## REST API

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/performance` | Combined summary of all profilers |
| `GET /api/v1/performance/frames` | Frame budget stats |
| `GET /api/v1/performance/servos` | Servo latency stats |
| `GET /api/v1/performance/bus` | Event bus throughput |

When profiling is disabled, each endpoint returns `{"enabled": false}`.

---

## CLI: deskbot-profile

The `deskbot-profile` CLI runs the robot for a specified duration, collects all
profiling data, and writes a JSON report:

```bash
deskbot-profile --duration 10 --output -         # print to stdout
deskbot-profile --duration 30 --output profile.json
```
