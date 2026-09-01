# Configuration

DeskBot configuration is defined by `robot.config.AppSettings` and its nested
Pydantic settings models.

## Sources and precedence

The application supports:

- YAML configuration
- `.env`
- environment variables
- programmatic settings

Environment variables are the preferred deployment override mechanism.

Nested configuration uses `__`, for example:

```env
DESKBOT_FACE__THEME=cute
DESKBOT_DISPLAYS__BACKEND=mock
DESKBOT_API__PORT=8000
```

## Top-level settings

`AppSettings` currently contains:

- `env`
- `log_level`
- `timezone`
- `hardware`
- `config_file`
- `assets_dir`
- `use_mocks`
- `personality`
- `displays`
- `face`
- `servos`
- `audio`
- `microphone`
- `camera`
- `perception`
- `llm`
- `tts`
- `stt`
- `wakeword`
- `sounds`
- `memory`
- `vector_memory`
- `learning`
- `teaching`
- `preferences`
- `conversation`
- `tools`
- `plugins`
- `mqtt`
- `homeassistant`
- `api`
- `performance`
- `logging`

The generated API configuration endpoint masks sensitive values before
returning them.

## Common examples

```env
DESKBOT_HARDWARE=mock
DESKBOT_FACE__THEME=vector
DESKBOT_SERVOS__BACKEND=mock
DESKBOT_DISPLAYS__BACKEND=mock

DESKBOT_LLM__PROVIDER=ollama
DESKBOT_TTS__PROVIDER=piper
DESKBOT_STT__PROVIDER=whisper

# ElevenLabs cloud TTS (high-quality, requires API key)
DESKBOT_TTS__PROVIDER=elevenlabs
DESKBOT_TTS__ELEVENLABS__API_KEY=your-api-key
DESKBOT_TTS__ELEVENLABS__VOICE_ID=21m00Tcm4TlvDq8ikWAM
DESKBOT_TTS__ELEVENLABS__MODEL_ID=eleven_multilingual_v2
DESKBOT_WAKEWORD__PROVIDER=openwakeword

DESKBOT_API__ENABLED=true
DESKBOT_API__HOST=0.0.0.0
DESKBOT_API__PORT=8000
```

## Learning and teaching

On-device learning (`learning`) and the human-in-the-loop teaching loop
(`teaching`) are gated off by default. Enabling learning arms the experience
recorder and background training; teaching additionally requires learning to be
enabled.

```env
# Enable on-device learning (experience recording + background training)
DESKBOT_LEARNING__ENABLED=true
DESKBOT_LEARNING__STORE=sqlite                       # 'memory' (lost on restart) or 'sqlite'
DESKBOT_LEARNING__USE_MULTIMODAL=true                # 570-D trainable MultimodalEncoder
DESKBOT_LEARNING__MULTIMODAL_HISTORY_LENGTH=5

# Enable the gesture-triggered teaching loop (requires learning enabled)
DESKBOT_TEACHING__ENABLED=true
DESKBOT_TEACHING__FEEDBACK_WINDOW_S=5.0              # max transition age for feedback attribution
DESKBOT_TEACHING__STALENESS_S=30.0                   # staleness bound (stored, not currently enforced)
DESKBOT_TEACHING__PRACTICE_EPSILON=0.2               # reserved (not currently wired; policy uses own decay)
DESKBOT_TEACHING__COOLDOWN_S=0.2                     # safety-gate cooldown between same action
DESKBOT_TEACHING__MIN_EXPERIENCES_FOR_PRACTICE=64    # below this, practice falls back to demonstration
```

Teaching is a *context flag*, never a `RobotState` one-hot slot, so enabling it
does not change `STATE_SIZE` (91 / 570 multimodal). See
[Learning](../modules/learning.md), [Teaching Mode](../modules/teaching_mode.md),
and the [production learning architecture](../architecture/production-learning.md).

For the authoritative field definitions, defaults, validators, and descriptions,
see `robot.config`.

## Logging

The `logging` block (`DESKBOT_LOGGING__` prefix) controls the dashboard log
ring buffer and the default "noisy events" hide list. See
[Logging](../modules/logging.md) for how these are used.

```env
# How many recent log entries the /#/logs dashboard keeps in memory (FIFO; 10-10000):
DESKBOT_LOGGING__RING_BUFFER_CAPACITY=500
# Event type names hidden by default in the dashboard "Recent Events" feed
# (comma-separated; high-frequency events that fire every frame). Toggleable in the UI:
DESKBOT_LOGGING__NOISY_EVENTS=DisplayUpdated,LookRequested,BlinkRequested,ServoMoved,IdleTimeout,LookAroundAction,FaceDetected
```
