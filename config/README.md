# YAML configuration

DeskBot supports loading configuration from a YAML file in addition
to the existing `.env` / environment-variable system. The YAML file is
**strictly lower-priority** than environment variables, so you can use
the YAML for defaults and override individual fields in your shell.

## Enabling

Set the path to the YAML file via the `DESKBOT_CONFIG_FILE` environment
variable:

```bash
export DESKBOT_CONFIG_FILE=/home/antonio/robot/config.yaml
uv run deskbot
```

If the file doesn't exist, DeskBot silently falls back to `.env`.

## Schema

The YAML is a flat dict whose top-level keys map to the corresponding
nested settings class. Example:

```yaml
# config.yaml
displays:
  backend: circuitpython      # "mock" or "circuitpython" or "gc9a01"
  bus: 0
  device: 0
  width: 240
  height: 240
  fps: 30
  rotation: 0
  dc_pin: 25
  reset_pin: 24
  spi_hz: 32000000             # 32 MHz works on Pi 5 once colours are confirmed
  spi_mode: 0
  invert: true
  chunk_bytes: 4096

face:
  theme: vector               # vector | minimal | cute | pixel | retro_lcd | wireframe

servos:
  backend: gpio               # mock | gpio | pca9685
  gpio:
    frequency: 50
    pins:
      pan: 12
      tilt: 13
      left_arm: 18
      right_arm: 19

microphone:
  input_device: default
  sample_rate: 16000
  channels: 1
  frame_ms: 30

camera:
  device: 0
  width: 640
  height: 480
  fps: 30

hardware: real               # mock | real - switches in real USB drivers
env: development             # development | testing | production
log_level: INFO              # DEBUG | INFO | WARNING | ERROR

llm:
  provider: mock              # mock | openai | ollama | custom
  model: gpt-4o-mini
  api_key: ""                 # or DESKBOT_LLM__API_KEY=...
  base_url: ""                # or DESKBOT_LLM__BASE_URL=http://localhost:1234/v1
  temperature: 0.7
  max_tokens: 256
  timeout_s: 15.0

tts:
  provider: mock              # mock | piper | elevenlabs | openai | espeak
  voice: default
  rate: 1.0

stt:
  provider: mock              # mock | whisper | vosk | google
  model: base
  language: en

wakeword:
  provider: mock              # mock | openwakeword | porcupine | snowboy
  phrase: hey deskbot
  threshold: 0.5

personality:
  curiosity: 0.7
  energy: 0.6
  shyness: 0.3
  friendliness: 0.8
  playfulness: 0.7
```

## Precedence

When the same field is set in multiple places, the highest-priority
source wins. From lowest to highest:

1. **Init kwargs** (`AppSettings(field=value)`)
2. **YAML** (`DESKBOT_CONFIG_FILE=...`)
3. **`.env`** (`DESKBOT_...` lines in `.env`)
4. **Environment** (`export DESKBOT_...`)

Example - YAML sets `displays.backend: circuitpython`, env sets
`DESKBOT_DISPLAYS__BACKEND=gc9a01`, the env value wins.

## Validation

The YAML is parsed by `pydantic-settings` and validated against the
typed settings classes. An invalid YAML value (e.g.
`displays.fps: -5`) raises a clear `ValidationError` at startup, not a
silent runtime failure.

## See also

- `.env.example` - exhaustive list of every supported env var
- `docs/wiring.md` - pin assignments and hardware configuration
