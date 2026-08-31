# DeskBot

> A modular desktop companion robot for Raspberry Pi - animated face, body language, perception, speech, LLM
> conversation, on-device learning, and a web API.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-Ruff-orange.svg)](https://docs.astral.sh/ruff/)
[![MyPy](https://img.shields.io/badge/types-MyPy%20strict-blue.svg)](https://mypy.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/well-it-wasnt-me/RON)

DeskBot is a hardware-agnostic desktop companion robot built around a Raspberry Pi. Its architecture separates robot
behavior from hardware, so the same application runs against real hardware, mock implementations, or a headless
simulation - meaning development and testing happen on any workstation before deployment to a Pi.

## Quick start

```bash
git clone https://github.com/well-it-wasnt-me/RON.git
cd RON
uv sync --all-extras          # or: pip install -e ".[dev]"
deskbot-simulate              # runs with mock hardware - no Pi needed
```

## Features

| Subsystem         | What it does                                                                                                  |
|-------------------|---------------------------------------------------------------------------------------------------------------|
| **Face**          | Animated eyes, mouth, emotions, six themes, overlays on a GC9A01 round TFT                                    |
| **Body language** | Servo-driven head pan/tilt and arm gestures                                                                   |
| **Behavior**      | State machine: idle -> curious -> listening -> thinking -> speaking                                           |
| **Perception**    | USB or RTSP camera + YuNet face detection, gaze tracking, event-driven reactions                              |
| **Conversation**  | OpenAI/Ollama LLM with streaming, tool calling, memory, and persistence                                       |
| **Speech**        | Whisper STT; Piper/eSpeak/OpenAI/ElevenLabs TTS; openWakeWord detection; reactive sound effects               |
| **Learning**      | On-device neural network - experience recording, world model, action learning (Q-policy), preference learning |
| **Teaching**      | Human-in-the-loop teaching mode - demonstrate gestures, give spoken feedback, watch Q-values learn on device  |
| **API**           | FastAPI REST + WebSocket event stream with browser dashboards                                                 |
| **Integration**   | MQTT bridge, Home Assistant Auto Discovery, Telegram bridge, plugin system via entry points                   |
| **Resilience**    | Graceful degradation - hardware failures fall back to mocks, never crash                                      |

## Architecture

A central **event bus** decouples subsystems.

Components publish immutable events and subscribe to what they care about.

Hardware is abstracted behind protocol interfaces:

* `Display`
* `ServoController`
* `AudioOutput`
* `Microphone`
* `Camera`
* `LLM`
* `EventBus`

so behavior code never imports a concrete driver.

-> Full [architecture overview](docs/architecture/overview.md)

## Configuration

YAML, `.env`, or environment variables (which take precedence). Nested fields use `__`:

```env
DESKBOT_HARDWARE=real
DESKBOT_DISPLAYS__BACKEND=circuitpython
DESKBOT_SERVOS__BACKEND=gpio
DESKBOT_LLM__PROVIDER=ollama
DESKBOT_TTS__PROVIDER=piper
DESKBOT_STT__PROVIDER=whisper
```

->
Full [configuration reference](docs/reference/config.md) · [example.yaml](config/example.yaml) · [.env.example](.env.example)

## API & dashboard

```text
http://<PI_IP>:8000/               # main dashboard
http://<PI_IP>:8000/calibration/   # servo & display calibration
http://<PI_IP>:8000/settings/      # hardware test page
http://<PI_IP>:8000/config         # configuration validator
http://<PI_IP>:8000/learning/      # learning dashboard
http://<PI_IP>:8000/teaching/     # teaching-loop dashboard
http://<PI_IP>:8000/docs           # OpenAPI documentation
```

WebSocket: `ws://<PI_IP>:8000/api/v1/ws/events`

-> Full [REST API reference](docs/reference/api.md)

## Documentation

Built with MkDocs Material - run `make docs-serve` for a local searchable site.

| Section                                                    | Key pages                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Getting Started](docs/getting-started/developer-setup.md) | [Developer setup](docs/getting-started/developer-setup.md) · [Wiring](docs/getting-started/wiring.md) · [Deployment](docs/getting-started/deployment.md) · [Contributing](docs/getting-started/contributing.md)                                                                                                                                                                                                                                         |
| [Architecture](docs/architecture/overview.md)              | [Overview](docs/architecture/overview.md) · [Learning](docs/architecture/learning.md) · [Audio](docs/architecture/audio.md) · [Lifecycle](docs/architecture/lifecycle.md) · [Production Learning](docs/architecture/production-learning.md)                                                                                                                                                                                                             |
| [Modules](docs/modules/face.md)                            | [Face](docs/modules/face.md) · [Body language](docs/modules/body-language.md) · [Behavior](docs/modules/behavior.md) · [Perception](docs/modules/perception.md) · [Conversation](docs/modules/conversation.md) · [Speech](docs/modules/speech.md) · [Learning](docs/modules/learning.md) · [Events](docs/modules/events.md) · [Interfaces](docs/modules/interfaces.md) · [Plugins](docs/modules/plugins.md) · [Teaching](docs/modules/teaching_mode.md) |
| [CLI](docs/cli/index.md)                                   | [Overview](docs/cli/index.md) · [Interactive TUI](docs/cli/interactive.md)                                                                                                                                                                                                                                                                                                                                                                              |
| [Reference](docs/reference/config.md)                      | [Config](docs/reference/config.md) · [REST API](docs/reference/api.md) · [Errors](docs/reference/errors.md) · [Hardware](docs/reference/hardware.md) · [Performance](docs/reference/performance.md)                                                                                                                                                                                                                                                     |

## License

Released under the [MIT License](LICENSE).
