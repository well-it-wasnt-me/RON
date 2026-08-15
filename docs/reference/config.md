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
- `conversation`
- `tools`
- `plugins`
- `mqtt`
- `api`

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
DESKBOT_WAKEWORD__PROVIDER=energy

DESKBOT_API__ENABLED=true
DESKBOT_API__HOST=0.0.0.0
DESKBOT_API__PORT=8000
```

For the authoritative field definitions, defaults, validators, and descriptions,
see `robot.config`.
