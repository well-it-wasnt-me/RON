# Lifecycle architecture

DeskBot's lifecycle management ensures the application starts and stops
cleanly, and that hardware failures never crash the robot.

---

## Startup sequence

`DeskBotApp.build()` constructs all subsystems, then `_on_startup()` runs
the ordered startup hooks:

```mermaid
flowchart TD
    Build["DeskBotApp.build()"] --> Sub["Construct subsystems<br/>(display, servos, audio, AI, perception...)"]
    Sub --> Start["_on_startup()"]
    Start --> S1["Start lifecycle task group"]
    S1 --> S2["Load learning checkpoint (if enabled)"]
    S2 --> S3["Start face animator"]
    S3 --> S4["Start perception service (if enabled)"]
    S4 --> S5["Start conversation service (if enabled)"]
    S5 --> S6["Start MQTT/HA bridges (if enabled)"]
    S6 --> S7["Start API server (if enabled)"]
    S7 --> S8["Load & start plugins (if enabled)"]
    S8 --> Running["Robot is RUNNING"]
```

Every step is wrapped in defensive `try`/`except` (via `contextlib.suppress`
or `safe_init`). If a component fails to start, the robot continues with a
mock or no-op fallback — it never crashes.

---

## Shutdown sequence

`_on_shutdown()` runs shutdown hooks in reverse order:

```mermaid
flowchart TD
    Stop["_on_shutdown()"] --> D1["Stop API server"]
    D1 --> D2["Stop MQTT/HA bridges"]
    D2 --> D3["Stop conversation service"]
    D3 --> D4["Stop perception service"]
    D4 --> D5["Stop face animator"]
    D5 --> D6["Stop learning service (join daemon thread)"]
    D6 --> D7["Close stores (SQLite connections)"]
    D7 --> D8["Cancel lifecycle task group"]
    D8 --> D9["Unsubscribe from event bus"]
    D9 --> Stopped["Robot is STOPPED"]
```

Shutdown is idempotent — calling it twice is safe. All background tasks are
cancelled and joined before the process exits.

---

## Graceful degradation

The degradation system is the robot's resilience mechanism. When a hardware
backend fails to initialise, `safe_init()` catches the exception, logs a
`WARNING`, records a `DegradationEntry`, and returns a mock fallback.

### How it works

```mermaid
flowchart LR
    Factory["Factory call<br/>(e.g. GC9A01Display)"] -->|"raises"| Catch["safe_init catches"]
    Catch --> Log["Log WARNING"]
    Log --> Record["Record DegradationEntry"]
    Record --> Fallback["Return MockDisplay"]
    Fallback --> Robot["Robot continues"]
```

### Components with fallbacks

| Component | Fallback | When |
|-----------|----------|------|
| Display | `MockDisplay` | GC9A01/CircuitPython init fails |
| Servos | `MockServoBus` | GPIO/PCA9685 init fails |
| Audio output | `MockAudioOutput` | USB/Bluetooth init fails |
| Microphone | (None — disabled) | USB microphone init fails |
| Camera | `MockCamera` | USB camera init fails |
| LLM | `MockLLM` | OpenAI/Ollama connection fails |
| STT | `MockSTT` | Whisper init fails |
| TTS | `MockTTS` | Provider init fails |
| Wake word | `NullWakeWordChecker` | Provider dependency missing |

### Health monitoring

The `GET /api/v1/health` endpoint returns the overall status (`"ok"` or
`"degraded"`) and per-component degradation details, so external monitoring
tools can detect when the robot is running with fallback hardware.

---

## Lifecycle states

The `Lifecycle` manager transitions through these states:

| State | Description |
|-------|-------------|
| `NEW` | Not yet started |
| `STARTING` | Startup hooks are running |
| `RUNNING` | All startup hooks complete; task group active |
| `STOPPING` | Shutdown hooks are running |
| `STOPPED` | All shutdown hooks complete; resources released |

The lifecycle owns the root `anyio` task group. Components subscribe to the
event bus during startup and unsubscribe during shutdown — the manager
guarantees both happen exactly once.
