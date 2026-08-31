# DeskBot

**A modular desktop companion robot powered by Raspberry Pi.**

DeskBot is an asynchronous, event-driven Python application that brings a
small robot to life on your desk. It combines an animated face on a circular
TFT display, expressive servo-driven body language, camera-based perception,
speech recognition and synthesis, and a conversational AI stack - all
connected through an internal event bus.

![DeskBot logo](assets/deskbot_logo.png){: .center}

---

## Quick start

```bash
# Clone the repository
git clone https://github.com/well-it-wasnt-me/RON.git
cd RON

# Create a virtual environment (Python 3.12+)
python -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"

# Run in simulation mode (no hardware needed)
deskbot-simulate
```

See the [Developer Setup](getting-started/developer-setup.md) guide for full instructions,
including hardware wiring and deployment.

## What's inside?

| Subsystem | Description |
|-----------|-------------|
| **Face engine** | Animated eyes, mouth, and expressions on a GC9A01 round display |
| **Body language** | Servo choreography for head pan/tilt and arm gestures |
| **Behavior engine** | State machine (idle → curious → listening → thinking → speaking) |
| **Perception** | YuNet face detection via USB camera |
| **Speech** | Whisper STT, Piper/eSpeak/OpenAI/ElevenLabs TTS, wake-word detection |
| **Conversation** | Ollama or OpenAI LLM with streaming, tools, and persistence |
| **Learning** | On-device neural network: experience recording, world model, action learning (Q-policy), preference learning |
| **Preferences** | Learns user name, volume, pace, formality, humour, and more |
| **REST API** | FastAPI server with WebSocket event streaming |
| **Teaching** | Human-in-the-loop teaching mode - demonstrate gestures, give spoken feedback, learn on device |
| **MQTT bridge** | Publishes events and receives commands over MQTT |
| **Telegram bridge** | Chat with the robot from Telegram |
| **Home Assistant** | MQTT Auto Discovery for native HA integration |
| **Plugin system** | Extend DeskBot via Python entry points |
| **Lifecycle** | Graceful degradation — robot never crashes on hardware failure |

## Architecture

DeskBot uses a central **event bus** to decouple subsystems. Components
publish immutable dataclass events and subscribe to the event types they
care about. Hardware is abstracted behind protocol interfaces so the
same application code runs on a Pi or in simulation.

Read the full [Architecture Overview](architecture/overview.md).

## Documentation

- **Getting Started**: [Developer Setup](getting-started/developer-setup.md) · [Wiring Guide](getting-started/wiring.md) · [Deployment](getting-started/deployment.md) · [Contributing](getting-started/contributing.md)
- **Architecture**: [Overview](architecture/overview.md) · [Learning](architecture/learning.md) · [Production Learning](architecture/production-learning.md) · [Audio](architecture/audio.md) · [Lifecycle](architecture/lifecycle.md)
- **Expression**: [Face](modules/face.md) · [Animations](modules/face-animations.md) · [Eye Engine](modules/eye-engine.md) · [Body Language](modules/body-language.md) · [Animation](modules/animation.md)
- **Behavior**: [Behavior Engine](modules/behavior.md) · [Behavior Library](modules/behavior-library.md) · [Perception](modules/perception.md)
- **Conversation & AI**: [Conversation](modules/conversation.md) · [Speech](modules/speech.md) · [Tools](modules/tools.md) · [Preferences](modules/preferences.md) · [Vector Memory](modules/vector-memory.md)
- **Learning**: [Local Brain](modules/learning.md) · [Teaching Mode](modules/teaching_mode.md)
- **Infrastructure**: [Events](modules/events.md) · [Interfaces](modules/interfaces.md) · [Lifecycle](modules/lifecycle.md) · [Services](modules/services.md) · [Simulation](modules/simulation.md) · [Performance](modules/performance.md)
- **Hardware**: [Servos](modules/servos.md)
- **Integration**: [Plugins](modules/plugins.md) · [MQTT Bridge](modules/mqtt.md) · [Home Assistant](modules/home-assistant.md) · [Telegram Bridge](modules/telegram.md)
- **CLI**: [Overview](cli/index.md) · [Interactive CLI](cli/interactive.md)
- **Reference**: [Config](reference/config.md) · [REST API](reference/api.md) · [Errors](reference/errors.md) · [Events](reference/events.md) · [Hardware](reference/hardware.md) · [Performance](reference/performance.md)

## Contributing

We welcome contributions! See [Contributing](getting-started/contributing.md) for guidelines.

## License

DeskBot is released under the [MIT License](https://github.com/well-it-wasnt-me/RON/blob/main/LICENSE).
