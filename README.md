# DeskBot

> A modular desktop companion robot for Raspberry Pi, with an animated face, expressive body language, perception, speech, LLM conversation, learning, and a web API.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-Ruff-orange.svg)](https://docs.astral.sh/ruff/)
[![MyPy](https://img.shields.io/badge/types-MyPy%20strict-blue.svg)](https://mypy.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

DeskBot is an open, hardware-agnostic desktop companion robot designed around a Raspberry Pi.

Its architecture separates robot behavior from hardware and infrastructure, allowing the same application to run against real hardware, mock implementations, or a headless simulation environment.

DeskBot currently combines:

* **Face**: animated eyes, eyebrows, mouth, emotions, themes, and overlays.
* **Body**: expressive head and arm movement through servo-based body language.
* **Perception**: camera input, face detection, gaze tracking, and perception-driven reactions.
* **Conversation**: speech recognition, wake-word detection, LLM integration, memory, and text-mode interaction.
* **Speech**: multiple STT and TTS providers.
* **Learning**: experience recording, replay, memory, tensor operations, and learning workflows.
* **Remote control**: FastAPI, WebSockets, and a browser dashboard.
* **Simulation**: mock hardware and headless operation for development and CI.

The application is designed to run **without physical hardware**, so development and testing can happen on a normal Linux or macOS workstation before deployment to a Raspberry Pi.

---

## Speech

Available speech components include:

### Text-to-speech

* OpenAI TTS
* ElevenLabs TTS
* Piper TTS
* eSpeak-NG TTS
* Mock TTS

### Speech-to-text

* Whisper
* Mock STT

### Audio and wake word

* openWakeWord integration
* Mock audio backends

Audio providers are abstracted behind interfaces so the conversation pipeline can be tested without physical audio hardware.

---

## Perception

The perception subsystem supports:

* USB cameras
* OpenCV
* YuNet face detection
* OpenCV cascade fallback
* Headless/null perception for CI
* Adaptive perception scan intervals
* Smooth gaze tracking
* Face-detection-driven robot reactions
* Event-driven perception behavior

Perception is separated from robot behavior. Detected events are published through the event bus and consumed by interested subsystems.

---

## Learning

DeskBot contains an emerging learning subsystem intended to support experience-based robot behavior.

Current components include:

* Experience recording
* State encoding
* Working memory
* Replay buffers
* Episodic-memory integration
* Tensor operations
* Learning/training workflows
* Learning evaluation
* Learning-state export/reset
* Learning-related CLI tools

The learning system is still under active development and should be considered experimental.

The architectural direction is to allow robot experiences to be captured independently from the eventual learning algorithm,
making it possible to experiment with different models and training approaches without changing the rest of the robot stack.

---

## Architecture

DeskBot is deliberately divided into independent layers.

```text
                         ┌───────────────────────┐
                         │      DeskBotApp       │
                         │ application lifecycle │
                         └───────────┬───────────┘
                                     │
             ┌───────────────────────┼───────────────────────┐
             │                       │                       │
             ▼                       ▼                       ▼
      ┌─────────────┐        ┌─────────────┐        ┌──────────────┐
      │  Behavior   │        │     AI      │        │  Perception  │
      │   Engine    │        │ Conversation│        │   / Camera   │
      └──────┬──────┘        └──────┬──────┘        └──────┬───────┘
             │                      │                      │
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │      Event Bus      │
                         │   async pub/sub     │
                         └─────────┬───────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
       ┌────────────┐      ┌──────────────┐      ┌──────────────┐
       │ Face Engine│      │ Body Language│      │ Speech / TTS │
       │            │      │    Engine    │      │    / STT     │
       └─────┬──────┘      └──────┬───────┘      └──────────────┘
             │                    │
             ▼                    ▼
       ┌────────────┐      ┌─────────────┐
       │  Display   │      │   Servos    │
       │ GC9A01/mock│      │ GPIO/mock/… │
       └────────────┘      └─────────────┘

                         ┌─────────────────────┐
                         │      Learning       │
                         │ experience / memory │
                         │ replay / training   │
                         └─────────────────────┘
```

The important architectural rule is:

> **Application logic should not depend directly on hardware.**

Hardware and infrastructure boundaries are represented by protocols such as:

* `Display`
* `ServoController`
* `AudioOutput`
* `Microphone`
* `Camera`
* `LLM`
* `StreamingLLM`
* `SpeechToText`
* `EventBus`

This allows the same behavior to run against real hardware, mocks, or simulation.

See [`docs/architecture/overview.md`](docs/architecture/overview.md) for the detailed architecture.

---

## Configuration

DeskBot supports YAML configuration, `.env` files, and environment-variable overrides. Environment variables take precedence over YAML.

Start from `config/example.yaml` or set environment variables directly:

```env
DESKBOT_HARDWARE=real
DESKBOT_DISPLAYS__BACKEND=circuitpython
DESKBOT_SERVOS__BACKEND=gpio
DESKBOT_LLM__PROVIDER=openai
DESKBOT_TTS__PROVIDER=piper
DESKBOT_STT__PROVIDER=whisper
```

Nested fields use `__` as a separator:

```env
DESKBOT_DISPLAYS__BACKEND=circuitpython
DESKBOT_FACE__THEME=vector
DESKBOT_API__PORT=8000
```

For the complete configuration reference, see [Configuration](docs/reference/config.md).

---

## API and dashboard

The optional API layer provides REST endpoints, a WebSocket event stream,
and a browser dashboard.

```text
http://<PI_IP>:8000/               # main dashboard
http://<PI_IP>:8000/calibration/   # servo and display calibration
http://<PI_IP>:8000/settings/      # hardware test page
http://<PI_IP>:8000/docs           # OpenAPI documentation
```

The WebSocket streams operator-facing events by default. Connect a client
to `ws://<PI_IP>:8000/api/v1/ws/events` to receive a live feed of state
changes, emotions, face detections, and wake words.

For the full API reference, see [REST API](docs/reference/api.md).

---


# Documentation

Documentation is built with MkDocs:

```bash
make docs         # build
make docs-serve   # serve with live reload
```

Key documentation:

* [Architecture overview](docs/architecture/overview.md)
* [Wiring guide](docs/wiring.md)
* [Developer setup](docs/developer-setup.md)
* [Configuration reference](docs/reference/config.md)
* [Hardware reference](docs/reference/hardware.md)
* [Deployment](docs/deployment.md)
* [Contributing](docs/contributing.md)
* [Events](docs/modules/events.md)
* [Audio architecture](docs/audio-architecture.md)
* [Learning architecture](docs/architecture-learning.md)

---

# License

RON DeskBot is released under the MIT License. See [LICENSE](LICENSE).
