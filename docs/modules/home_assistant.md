# Home Assistant Integration

DeskBot can appear as a native device in Home Assistant via MQTT Auto
Discovery. When enabled, the bridge publishes discovery payloads that
Home Assistant automatically picks up, creating entities for the robot
without any manual YAML configuration.

## Overview

The Home Assistant bridge:

- **Publishes** MQTT Auto Discovery configs for 5 entities (2 selects, 3 sensors).
- **Subscribes** to 2 command topics so HA automations can control the robot.
- Reports online/offline status via an availability topic with LWT.
- Uses `paho-mqtt` v2.1+ (same dependency as the MQTT bridge).

Both the MQTT bridge and the Home Assistant bridge can run simultaneously
on the same MQTT broker.

## Configuration

Home Assistant integration is **opt-in** and disabled by default:

```bash
DESKBOT_HOMEASSISTANT__ENABLED=true
DESKBOT_HOMEASSISTANT__HOST=homeassistant.local
DESKBOT_HOMEASSISTANT__PORT=1883
DESKBOT_HOMEASSISTANT__USERNAME=deskbot
DESKBOT_HOMEASSISTANT__PASSWORD=secret
DESKBOT_HOMEASSISTANT__DEVICE_ID=deskbot
DESKBOT_HOMEASSISTANT__DEVICE_NAME=DeskBot
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DESKBOT_HOMEASSISTANT__ENABLED` | `false` | Enable or disable the HA bridge |
| `DESKBOT_HOMEASSISTANT__HOST` | `homeassistant.local` | MQTT broker host (typically your HA instance) |
| `DESKBOT_HOMEASSISTANT__PORT` | `1883` | MQTT broker port |
| `DESKBOT_HOMEASSISTANT__USERNAME` | `""` | Broker username (empty = no auth) |
| `DESKBOT_HOMEASSISTANT__PASSWORD` | `""` | Broker password |
| `DESKBOT_HOMEASSISTANT__DISCOVERY_PREFIX` | `homeassistant` | HA MQTT discovery prefix |
| `DESKBOT_HOMEASSISTANT__DEVICE_ID` | `deskbot` | HA device identifier |
| `DESKBOT_HOMEASSISTANT__DEVICE_NAME` | `DeskBot` | HA device display name |
| `DESKBOT_HOMEASSISTANT__DEVICE_MANUFACTURER` | `DeskBot Contributors` | HA device manufacturer |
| `DESKBOT_HOMEASSISTANT__DEVICE_MODEL` | `Desktop Companion Robot` | HA device model |
| `DESKBOT_HOMEASSISTANT__QOS` | `1` | MQTT QoS level |

## Entities created

When the bridge starts, it publishes MQTT Auto Discovery configs for
the following Home Assistant entities:

### Select entities

| Entity ID | Options | Description |
|-----------|---------|-------------|
| `select.deskbot_state` | idle, curious, listening, thinking, speaking, sleeping | Robot state |
| `select.deskbot_emotion` | neutral, happy, curious, thinking, sleepy, embarrassed, excited, sad, surprised, angry | Robot emotion |

### Sensor entities

| Entity ID | Description |
|-----------|-------------|
| `sensor.deskbot_wake_word` | Last wake word detection |
| `sensor.deskbot_face_detected` | Face detection status |
| `sensor.deskbot_sound_effect` | Last sound effect played |

## Controlling the robot from Home Assistant

### Change emotion

In an HA automation or developer tools:

```yaml
service: mqtt.publish
data:
  topic: homeassistant/select/deskbot/emotion/set
  payload: '{"emotion": "happy", "intensity": 0.8}'
```

### Change state

```yaml
service: mqtt.publish
data:
  topic: homeassistant/select/deskbot/state/set
  payload: '{"state": "curious"}'
```

### Example automation

Trigger the robot to look curious when a person is detected by HA:

```yaml
alias: "DeskBot - Look curious when person detected"
trigger:
  - platform: state
    entity_id: binary_sensor.hallway_motion
    to: "on"
action:
  - service: mqtt.publish
    data:
      topic: homeassistant/select/deskbot/emotion/set
      payload: '{"emotion": "curious"}'
  - service: mqtt.publish
    data:
      topic: homeassistant/select/deskbot/state/set
      payload: '{"state": "curious"}'
```

## Installation

```bash
pip install paho-mqtt>=2.1
```

Or:

```bash
pip install deskbot[mqtt]
# or
pip install deskbot[homeassistant]
```

If `paho-mqtt` is not installed and HA is enabled, DeskBot logs a
warning and continues without the bridge - the robot runs normally.
