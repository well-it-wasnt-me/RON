# Performance Profiling

DeskBot includes lightweight performance profiling that lets you measure and
monitor frame budget compliance, servo latency, event bus throughput, and
identify bottlenecks.

## Overview

The profiling system has three components:

| Profiler | What it measures | Location |
|----------|-----------------|----------|
| FrameProfiler | Frame rendering time vs. budget | `robot.performance.frame_profiler` |
| ServoProfiler | Servo command latency | `robot.performance.servo_profiler` |
| BusProfiler | Event bus throughput & processing time | `robot.performance.bus_profiler` |

## Configuration

Performance profiling is controlled via `PerformanceConfig` (environment
prefix: `DESKBOT_PERFORMANCE__`):

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `True` | Master switch. When `False`, all profilers are no-ops. |
| `frame_profiling` | `True` | Enable frame budget monitoring. |
| `servo_profiling` | `True` | Enable servo latency profiling. |
| `bus_profiling` | `True` | Enable event bus throughput profiling. |
| `bus_sample_rate` | `0.1` | Fraction of bus events to sample (10% = 1 in 10). |
| `report_interval_frames` | `100` | Publish `FrameStatsReport` event every N frames. |

Example environment variables:

```bash
# Disable all profiling
DESKBOT_PERFORMANCE__ENABLED=false

# Disable bus profiling only
DESKBOT_PERFORMANCE__BUS_PROFILING=false

# Sample 25% of bus events
DESKBOT_PERFORMANCE__BUS_SAMPLE_RATE=0.25
```

## Frame Budget

The **frame budget** is the maximum time allowed per frame to hit the target
FPS. At 30 FPS, the budget is 33.33 ms per frame. A frame that exceeds this
budget is considered **dropped**.

### Reading the stats

```
FrameStats:
  target_fps: 30          # Configured target
  actual_fps: 28.5        # Measured average
  avg_frame_time_ms: 35.1 # Average frame duration
  p50_frame_time_ms: 33.2 # 50th percentile
  p95_frame_time_ms: 38.0 # 95th percentile
  p99_frame_time_ms: 42.5 # 99th percentile
  dropped_frames: 12      # Frames exceeding budget
  total_frames: 300       # Total frames recorded
```

- **actual_fps ≈ target_fps**: System is healthy.
- **actual_fps << target_fps**: Rendering is too slow; check p95/p99 for
  outliers.
- **dropped_frames > 0**: Some frames exceeded the budget. A few dropped
  frames per 1000 is normal; sustained drops indicate a problem.

### Pi 5 Baselines

On a Raspberry Pi 5 with the GC9A01 display:

| Metric | Expected range |
|--------|----------------|
| avg_frame_time_ms | 15–25 ms |
| p95_frame_time_ms | 25–33 ms |
| actual_fps | 28–30 fps |

## Servo Latency

The servo profiler measures the time from when `servo.move_to()` is called to
when the `ServoMoved` event is published on the bus. For real servos (GPIO /
PCA9685), this includes physical movement time. For mock servos, latency
should be near-zero.

### Reading the stats

```json
{
  "pan": {"avg_ms": 12.5, "p50_ms": 10.0, "p95_ms": 25.0, "count": 100},
  "tilt": {"avg_ms": 8.3, "p50_ms": 7.5, "p95_ms": 15.0, "count": 100}
}
```

### Pi 5 Baselines

| Servo | avg_ms | p95_ms |
|-------|--------|--------|
| Mock | < 1 | < 2 |
| GPIO | 10–50 | 20–80 |
| PCA9685 | 10–50 | 20–80 |

## Event Bus Throughput

The bus profiler wraps `InMemoryEventBus.publish()` to measure event
processing time for a configurable sample of events.

### Reading the stats

```json
{
  "total_events": 5000,
  "sampled_events": 500,
  "sample_rate": 0.1,
  "events_per_second": 120.5,
  "avg_processing_time_ms": 0.5,
  "p50_processing_time_ms": 0.3,
  "p95_processing_time_ms": 1.2,
  "max_processing_time_ms": 5.0,
  "events_by_type": {"EmotionChanged": 200, "DisplayUpdated": 4500}
}
```

- **events_per_second**: Throughput of the event bus.
- **p95_processing_time_ms**: Most handlers complete within this time.
- **max_processing_time_ms**: Outlier events that took longest.

## REST API

All profiling data is exposed via the REST API:

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/performance` | Combined summary of all profilers |
| `GET /api/v1/performance/frames` | Frame budget stats |
| `GET /api/v1/performance/servos` | Servo latency stats |
| `GET /api/v1/performance/bus` | Event bus throughput |

Example:

```bash
curl http://localhost:8000/api/v1/performance/frames
```

When profiling is disabled, each endpoint returns `{"enabled": false}`.

## CLI: deskbot-profile

The `deskbot-profile` CLI command runs the robot for a specified duration,
collects all profiling data, and writes a JSON report:

```bash
# Run for 10 seconds and print to stdout
deskbot-profile --duration 10 --output -

# Run for 30 seconds and save to a file
deskbot-profile --duration 30 --output profile.json

# Use a custom config file
deskbot-profile --duration 10 --config-file config.yaml --output profile.json
```

## Overhead

When profiling is **disabled** (`DESKBOT_PERFORMANCE__ENABLED=false`):

- `FrameProfiler.record_frame()` returns immediately (single boolean check).
- `ServoProfiler` never subscribes to events.
- `BusProfiler` never wraps `publish()`.

When enabled, overhead is minimal:

- **FrameProfiler**: ~1 µs per frame (one `time.monotonic()` call + list append).
- **ServoProfiler**: ~0.5 µs per event (dict lookup + list append).
- **BusProfiler**: ~0.1 µs per event for unsampled events; ~5 µs for sampled.

All timing uses `time.monotonic()` (never `time.time()`).

Stats use **rolling windows** (last N entries), not unbounded accumulators, so
memory usage is constant regardless of runtime.
