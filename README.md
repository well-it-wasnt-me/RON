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

## What it can do

### Face and animation

DeskBot provides a display-independent face and animation system supporting:

* 30 FPS animated face rendering
* Multiple visual themes:
  * Vector
  * Minimal
  * Cute
  * Pixel
  * Retro LCD
  * Wireframe
* Emotion-driven facial state
* Blinking
* Gaze movement
* Speaking and thinking animations
* Animation timelines
* Easing functions
* Overlay rendering
* Display-independent rendering
* Circular 240×240 TFT support
* GC9A01 / CircuitPython display backend
* In-memory mock display for development and testing

The face system is intentionally separated from the physical display, allowing the renderer to be tested independently.

---

## Body language

DeskBot can coordinate multiple servos for expressive movement:

* Head pan
* Head tilt
* Left arm
* Right arm

Application code works with high-level body-language requests rather than directly manipulating GPIO pins.

Examples include:

* Looking toward a target
* Waving
* Celebrating
* Reacting to perception
* Expressive servo movement
* Coordinated multi-servo actions

### Servo backends

| Backend | Status |
| --- | --- |
| `mock` | Available |
| `gpio` | Available on Raspberry Pi |
| `pca9685` | Partial / work in progress |

The GPIO backend includes:

* Configurable angle ranges
* Continuous angle-to-PWM mapping
* Servo inversion
* Redundant-command suppression
* Explicit servo release
* Reassertion after release
* Range validation

Servo control is exposed through the hardware abstraction layer so behavior code does not depend on a particular GPIO implementation.

---

## Conversation and AI

DeskBot includes a conversational pipeline with:

* OpenAI LLM support
* Ollama local LLM support
* Streaming LLM responses
* SQLite conversation persistence
* Conversation memory
* Tool/function-call infrastructure
* Configurable personality parameters
* Text-mode interaction
* Conversation events through the event bus

The text interface is particularly useful when developing without a microphone or speaker.

```bash
uv run deskbot chat
```

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

* USB audio input/output
* Energy-based wake-word detection
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

The architectural direction is to allow robot experiences to be captured independently from the eventual learning algorithm, making it possible to experiment with different models and training approaches without changing the rest of the robot stack.

---

## Events

Subsystems communicate primarily through an asynchronous event bus.

Events are used for things such as:

* State changes
* Emotion changes
* Servo movement
* Face detection
* Speech recognition
* Idle timeouts
* Perception scans
* Robot reactions

This keeps components loosely coupled.

The event bus also supports explicit subscription and unsubscription, allowing behaviors and services to attach and detach cleanly during their lifecycle.

---

# Architecture

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

# Repository structure

```text
DeskBot/
├── assets/
│   ├── animations/        # animation assets
│   ├── eyes/              # some specific eye render
│   └── sounds/            # robot sound effects
│
├── web/                   # browser dashboard
│
├── config/
│   ├── example.yaml       # example configuration
│   └── README.md          # configuration notes
│
├── deploy/
│   └── systemd/           # Raspberry Pi service definitions
│
├── docs/
│   ├── architecture/      # system architecture
│   ├── modules/           # subsystem documentation
│   ├── reference/         # configuration and API reference
│   ├── developer-setup.md
│   ├── contributing.md
│   ├── roadmap.md
│   ├── wiring.md
│   └── architecture-learning.md
│
├── scripts/
│   └── install.sh         # installation helper
│
├── src/robot/
│   ├── ai/                # LLM, conversation and memory
│   ├── animation/         # timelines, easing and scheduling
│   ├── api/               # FastAPI and WebSocket API
│   ├── behavior/          # state machine and reactions
│   ├── behavior_library/  # reusable robot behaviors
│   ├── body_language/     # servo choreography
│   ├── cli/               # command-line tools
│   ├── events/            # event bus and event types
│   ├── face/              # face model, renderer, themes, emotions
│   ├── hardware/          # hardware implementations
│   ├── interfaces/        # hardware/service protocols
│   ├── learning/          # experience recording and learning
│   ├── lifecycle/         # lifecycle and graceful degradation
│   ├── perception/        # camera and face detection
│   ├── performance/       # profiling and instrumentation
│   ├── plugins/           # plugin infrastructure
│   ├── services/          # application services
│   ├── simulation/        # headless robot simulation
│   ├── speech/            # STT, TTS and wake word
│   └── utils/             # shared utilities
│
├── tests/
│   ├── fakes/             # fake implementations
│   ├── integration/       # integration tests
│   └── unit/              # isolated component tests
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── mkdocs.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Quick start

## Requirements

For development without hardware:

* Python 3.12+
* [`uv`](https://docs.astral.sh/uv/)
* Linux or macOS
* A POSIX-compatible shell

For Raspberry Pi deployment:

* Raspberry Pi 5
* Raspberry Pi OS 64-bit
* SPI enabled for the TFT
* GPIO access for servos
* Optional I2C, audio, and camera hardware depending on the features being used

---

## Installation

```bash
git clone <repository-url>
cd DeskBot

make install
```

`make install` uses `uv` and installs the project's development, runtime, hardware, AI, audio, TTS, vision, API, and documentation dependencies.

---

## Run without hardware

DeskBot can run with mock hardware.

```bash
make run
```

Useful commands include:

```bash
make doctor
make simulate
make eye-demo
make display-test
```

The application can also be launched directly:

```bash
uv run deskbot
```

---

# Configuration

DeskBot supports YAML configuration and environment-variable overrides.

Start from:

```text
config/example.yaml
```

Example:

```yaml
hardware: real

displays:
  backend: circuitpython
  width: 240
  height: 240
  fps: 30

face:
  theme: vector

servos:
  backend: gpio

llm:
  provider: openai

tts:
  provider: piper

stt:
  provider: whisper
```

Environment variables use nested `__` notation:

```bash
export DESKBOT_HARDWARE=real
export DESKBOT_DISPLAYS__BACKEND=circuitpython
export DESKBOT_SERVOS__BACKEND=gpio
export DESKBOT_LLM__PROVIDER=ollama
```

Environment variables take precedence over YAML configuration.

See [`docs/reference/config.md`](docs/reference/config.md) for the complete configuration reference.

---

# Raspberry Pi hardware

## Display

The current prototype uses:

* 1.28" 240×240 round GC9A01 TFT
* SPI
* GC9A01 support
* CircuitPython display backend

---

## Servos

The default servo mapping is:

| Function | BCM GPIO |
| --- | ---: |
| Head pan | 12 |
| Head tilt | 13 |
| Left arm | 18 |
| Right arm | 19 |

Servo configuration includes:

* Minimum and maximum angles
* Pulse-width configuration
* Inversion
* Backend selection
* Servo release behavior

**Do not power servos directly from the Raspberry Pi 5V rail.**

Use an appropriate external servo power supply and connect its ground to Raspberry Pi ground.

See [`docs/wiring.md`](docs/wiring.md) before connecting hardware.

---

## Camera and audio

USB camera, microphone, and speaker backends are available for the perception and conversation pipelines.

Hardware-specific setup belongs in the documentation rather than here.

---

# Development

The Makefile is the preferred interface for common development tasks.

```bash
make help
```

## Formatting and linting

```bash
make format
make lint
```

## Type checking

```bash
make typecheck
```

MyPy runs in strict mode.

## Tests

```bash
make test
```

The test suite is designed to run without physical hardware.

For coverage:

```bash
make coverage
```

The repository currently targets a minimum coverage threshold of **80%**.

## Full verification

Before opening a pull request:

```bash
make check
```

This runs the project's formatting/linting, type checking, and automated tests.

---

# Simulation and diagnostics

DeskBot includes several command-line tools.

| Command | Purpose |
| --- | --- |
| `deskbot` | Run the main application |
| `deskbot chat` | Interactive text chat |
| `deskbot-doctor` | Diagnose environment and configuration |
| `deskbot-simulate` | Run the robot stack with simulated hardware |
| `deskbot-eye-demo` | Demonstrate the eye animation engine |
| `deskbot-display-test` | Test the physical display |
| `deskbot-hardware-check` | Check hardware availability |
| `deskbot-face-test` | Exercise the face renderer |
| `deskbot-learning-status` | Inspect learning subsystem status |
| `deskbot-learning-train` | Run learning/training workflows |
| `deskbot-learning-evaluate` | Evaluate learning components |
| `deskbot-learning-reset` | Reset learning state |
| `deskbot-learning-export` | Export learning data |
| `deskbot-profile` | Run application profiling |

The simulation and mock implementations are particularly useful for developing behavior and animation code away from the physical robot.

---

# Text mode

Text mode lets you interact with DeskBot by typing instead of speaking.

It is useful for:

* Development without a microphone
* Development without speakers
* Debugging the LLM pipeline
* Debugging TTS
* Testing conversation behavior
* Running DeskBot on a normal workstation

Start it with:

```bash
uv run deskbot chat
```

Type a message and press Enter.

DeskBot processes the input through the normal conversation pipeline, including LLM processing, tool calling, and TTS where configured.

Commands include:

```text
/quit
/exit
/help
/clear
```

No microphone or wake word is required.

A fully mocked configuration can be used for development:

```bash
DESKBOT_HARDWARE=mock \
DESKBOT_AUDIO__BACKEND=mock \
DESKBOT_TTS__PROVIDER=mock \
uv run deskbot chat
```

---

# Learning and experience recording

The learning subsystem is currently experimental.

Experience recording listens to robot events and converts them into encoded experiences containing information such as:

* Robot state
* Emotional state
* Servo activity
* Face/perception information
* Speech events
* Timing and reward information

Recorded experiences can be stored in working memory and replay buffers and can optionally integrate with episodic memory.

The subsystem is deliberately connected through the event bus so learning can observe the robot without becoming tightly coupled to individual hardware implementations.

Useful commands include:

```bash
uv run deskbot-learning-status
uv run deskbot-learning-train
uv run deskbot-learning-evaluate
uv run deskbot-learning-reset
uv run deskbot-learning-export
```

The learning architecture is documented in:

[`docs/architecture-learning.md`](docs/architecture-learning.md)

Learning algorithms and policies are still evolving and should not yet be considered production-stable.

---

# API and dashboard

The optional API layer is implemented with FastAPI.

It provides endpoints for:

* Robot state
* Health
* Conversation
* Perception
* Speech
* Emotion
* Configuration
* WebSocket event streaming

The browser dashboard is located under:

```text
web/
```

The API specification is reachable via:

```text
http://{DESKBOT_IP:8000}/docs
```

Docker support is available:

```bash
docker compose up --build
```

---

# Deployment

A systemd service definition is provided for Raspberry Pi deployment:

```text
deploy/systemd/deskbot.service
```

Docker support is provided through:

```text
Dockerfile
docker-compose.yml
```

Recommended deployment models:

| Environment             | Recommended approach          |
|-------------------------|-------------------------------|
| Development workstation | `uv` + mock hardware          |
| CI                      | `uv` + mock/headless hardware |
| Raspberry Pi robot      | `uv` + systemd                |
| API/server deployment   | Docker Compose                |

---

# Extending DeskBot

DeskBot is designed so new hardware and providers can be added without modifying application logic.

## Add a hardware backend

1. Define or reuse the appropriate protocol in `src/robot/interfaces/`.
2. Implement the backend in `src/robot/hardware/`.
3. Register it with the relevant factory.
4. Add configuration support.
5. Add unit tests using fakes/mocks.
6. Add integration coverage where appropriate.

## Add a behavior

1. Define the behavior/action.
2. Connect it to the behavior engine or event bus.
3. Add the corresponding executor route if required.
4. Add unit tests.
5. Add integration tests when the behavior crosses subsystem boundaries.

## Add an event

1. Define the payload in `src/robot/events/events.py`.
2. Export it from `src/robot/events/__init__.py`.
3. Publish it from the relevant subsystem.
4. Subscribe to it where the behavior belongs.
5. Ensure subscriptions are removed during component teardown.
6. Add tests for both subscription and unsubscription.

See [`docs/developer-setup.md`](docs/developer-setup.md) and [`docs/contributing.md`](docs/contributing.md).

---

# Current status

DeskBot is in **active development**.

The following major areas are implemented:

* Hardware abstraction
* Mock hardware
* Robot state machine
* Event-driven behavior
* Face rendering
* Animation engine
* Servo body language
* GPIO servo backend
* Perception pipeline
* Face detection
* LLM conversation
* Conversation persistence
* STT/TTS abstraction
* Wake-word detection
* FastAPI API
* WebSocket events
* Browser dashboard
* Simulation/headless operation
* Experience recording
* Learning infrastructure
* Performance instrumentation
* Lifecycle handling
* Automated unit and integration testing

The current codebase has a substantial amount of automated test suite and is intended to remain runnable without physical hardware.

---

# Roadmap

The major areas of ongoing development include:

* Complete PCA9685 servo implementation
* Additional Raspberry Pi audio hardware
* Expanded semantic/vector memory
* Learning-system hardening
* Training and evaluation workflows
* Streaming speech and lip-sync improvements
* MQTT integration
* Home Assistant integration
* More robust hardware-failure recovery
* Performance optimization
* Expanded configuration UI
* Expanded public documentation
* Mechanical/chassis references

---

# Documentation

Documentation is built with MkDocs.

Build locally:

```bash
make docs
```

Serve with live reload:

```bash
make docs-serve
```

Important documentation:

* [`Architecture`](docs/architecture/overview.md)
* [`Learning architecture`](docs/architecture-learning.md)
* [`Developer setup`](docs/developer-setup.md)
* [`Configuration`](docs/reference/config.md)
* [`Hardware wiring`](docs/wiring.md)
* [`Audio architecture`](docs/audio-architecture.md)
* [`Deployment`](docs/deployment.md)
* [`Contributing`](docs/contributing.md)

---

# Design principles

DeskBot follows a small set of architectural rules:

1. **Application logic should not know about hardware.**
2. **Hardware implementations should be replaceable.**
3. **The face is the primary expression channel.**
4. **Body language should reinforce facial expression rather than replace it.**
5. **Every subsystem should be testable without physical hardware.**
6. **Independent subsystems should communicate through explicit interfaces and events.**
7. **Lifecycle management must cleanly attach and detach resources.**
8. **Configuration should determine infrastructure choices wherever practical.**
9. **Learning should observe and consume robot experience without tightly coupling itself to robot hardware.**
10. **Simulation should remain a first-class development environment.**

The goal is to build a robot platform rather than a pile of Raspberry Pi scripts.

---

# License

DeskBot is released under the MIT License.

See [`LICENSE`](LICENSE).
