# MQTT Bridge

DeskBot can publish its internal events to an MQTT broker and receive
commands from external systems (dashboards, Home Assistant, Node-RED,
custom scripts) via the MQTT bridge.

## Overview

The MQTT bridge:

- **Publishes** 12 event types to `deskbot/events/{EventType}` topics.
- **Subscribes** to 4 command topics under `deskbot/commands/`.
- Reports online/offline status via `deskbot/status` (with LWT).
- Uses paho-mqtt v2.1+ for the MQTT connection.

## Configuration

MQTT is **opt-in** and disabled by default. Enable it with environment
variables:

```bash
DESKBOT_MQTT__ENABLED=true
DESKBOT_MQTT__HOST=mqtt.example.com
DESKBOT_MQTT__PORT=1883
DESKBOT_MQTT__USERNAME=deskbot
DESKBOT_MQTT__PASSWORD=secret
DESKBOT_MQTT__TOPIC_PREFIX=deskbot
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DESKBOT_MQTT__ENABLED` | `false` | Enable or disable the MQTT bridge |
| `DESKBOT_MQTT__HOST` | `localhost` | MQTT broker hostname |
| `DESKBOT_MQTT__PORT` | `1883` | MQTT broker port |
| `DESKBOT_MQTT__USERNAME` | `""` | Broker username (empty = no auth) |
| `DESKBOT_MQTT__PASSWORD` | `""` | Broker password |
| `DESKBOT_MQTT__TOPIC_PREFIX` | `deskbot` | MQTT topic prefix |
| `DESKBOT_MQTT__KEEPALIVE` | `60` | Keepalive interval in seconds |
| `DESKBOT_MQTT__QOS` | `1` | QoS level (0, 1, or 2) |
| `DESKBOT_MQTT__PUBLISH_EVENTS` | `true` | Publish local events to MQTT |
| `DESKBOT_MQTT__SUBSCRIBE_COMMANDS` | `true` | Subscribe to command topics |
| `DESKBOT_MQTT__HEARTBEAT_INTERVAL` | `30` | Heartbeat interval in seconds |

## Event topics

All events are published as JSON to `deskbot/events/{EventType}`:

| Topic suffix | Event type | Example payload |
|-------------|-----------|----------------|
| `StateChanged` | Robot state change | `{"type":"StateChanged","previous":"idle","current":"curious"}` |
| `EmotionChanged` | Emotion transition | `{"type":"EmotionChanged","previous":"neutral","current":"happy","intensity":0.8}` |
| `WakeWordDetected` | Wake word trigger | `{"type":"WakeWordDetected","phrase":"hey deskbot","confidence":0.95}` |
| `SpeechRecognized` | Speech transcript | `{"type":"SpeechRecognized","text":"hello","confidence":0.9}` |
| `FaceDetected` | Face detection | `{"type":"FaceDetected","x":0.5,"y":0.3,"confidence":0.85}` |
| `BlinkRequested` | Blink request | `{"type":"BlinkRequested","left":true,"right":true,"speed":1.0}` |
| `ServoMoved` | Servo position | `{"type":"ServoMoved","name":"pan","angle":45.0}` |
| `SoundEffectPlayed` | Sound effect | `{"type":"SoundEffectPlayed","name":"greet","filename":"greet.wav"}` |
| `LLMTokenReceived` | LLM streaming token | `{"type":"LLMTokenReceived","token":"Hello","done":false}` |
| `RobotStarted` | Robot started | `{"type":"RobotStarted"}` |
| `RobotStopped` | Robot stopped | `{"type":"RobotStopped","reason":"shutdown"}` |
| `RobotError` | Error | `{"type":"RobotError","message":"...","component":"..."}` |

## Command topics

The bridge subscribes to command topics and publishes events on the
local event bus. Send JSON payloads to control the robot:

### Change emotion

```
Topic: deskbot/commands/emotion
Payload: {"emotion": "happy", "intensity": 0.8}
```

### Change state

```
Topic: deskbot/commands/state
Payload: {"state": "curious"}
```

### Trigger speech

```
Topic: deskbot/commands/speak
Payload: {"text": "Hello there!"}
```

### Move servo

```
Topic: deskbot/commands/servo
Payload: {"name": "pan", "angle": 45.0}
```

## Status topic

The bridge publishes online/offline status with LWT (Last Will and
Testament) to `deskbot/status`:

- `online` - when the broker connection is established
- `offline` - when the connection is lost (via LWT)

## Home Assistant integration

See [Home Assistant integration](home-assistant.md) for MQTT Auto
Discovery support, which allows DeskBot to appear as a native device
in Home Assistant without manual YAML configuration.

## Example: Node-RED flow

1. Install `node-red-contrib-mqtt-broker` or use an external broker.
2. Subscribe to `deskbot/events/#` to monitor all events.
3. Publish to `deskbot/commands/emotion` with `{"emotion": "happy"}` to
   make the robot smile.

## Installation

```bash
pip install paho-mqtt>=2.1
```

Or install with the MQTT extras:

```bash
pip install deskbot[mqtt]
```

If `paho-mqtt` is not installed and MQTT is enabled, DeskBot logs a
warning and continues without the MQTT bridge - the robot runs normally.
