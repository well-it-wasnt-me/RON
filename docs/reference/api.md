# REST API and WebSocket API

The FastAPI application is created by `robot.api.create_app()`.

The API is mounted under `/api/v1`. FastAPI also serves its generated
OpenAPI document and interactive documentation (`/docs`, `/redoc`) when the
API is running.

## Health

- `GET /api/v1/health` - Health check
- `GET /api/v1/version` - Version info

## State and status

- `GET /api/v1/state` - Current robot state
- `POST /api/v1/state` - Transition state
- `GET /api/v1/config` - Current configuration (sensitive values masked)
- `GET /api/v1/perception` - Perception status
- `GET /api/v1/audio` - Audio output status
- `GET /api/v1/conversation` - Conversation status

## Commands

- `POST /api/v1/speak` - Speak text (injected into the conversation pipeline)
- `POST /api/v1/speak-direct` - Speak text directly via TTS (bypasses STT and LLM)
- `POST /api/v1/emotion` - Set emotion

`/speak` injects text into the conversational pipeline. `/speak-direct`
bypasses STT and LLM and sends text directly to the configured TTS.

## Conversations

- `GET /api/v1/conversations` - List conversations
- `GET /api/v1/conversations/{conversation_id}` - Get a conversation
- `DELETE /api/v1/conversations/{conversation_id}` - Delete a conversation

These operate on the configured conversation store.

## Preferences

- `GET /api/v1/preferences` - List all preferences
- `GET /api/v1/preferences/{key}` - Get a specific preference
- `DELETE /api/v1/preferences/{key}` - Delete a preference

See [Preference Tracking](../modules/preferences.md).

## Learning

- `GET /api/v1/learning/status` - Learning service status
- `GET /api/v1/learning/preferences` - Learned preferences
- `GET /api/v1/learning/config` - Learning configuration
- `POST /api/v1/learning/train` - Force a training cycle

These return "not available" responses when learning is disabled. See
[Local Brain](../modules/learning.md). `/api/v1/learning/status` reports
`enabled` from `settings.learning.enabled` (not a hard-coded value).

## Teaching

The teaching loop (human demonstration + feedback) is exposed under
`/api/v1/teaching/*`. POST endpoints require the API key
(`DESKBOT_API__API_KEY`) when one is configured. See
[Teaching Mode](../modules/teaching_mode.md).

- `GET /api/v1/teaching/status` - Teaching-loop status (enabled, in_teaching_mode, session_id, mode, trigger_gesture, desired_action, total_experiences, min_experiences_for_practice)
- `GET /api/v1/teaching/transitions?limit=` - Recent transitions (1-256, default 20) with a conversation-free state summary (teaching_context / interaction_active / person_present / gesture), reward, feedback_source, interaction_id, teaching_session_id
- `POST /api/v1/teaching/feedback` - Submit explicit human feedback `{polarity, magnitude, source, text}`; attributes to the most-recent eligible real transition. Returns `{attributed, transition_id?, delta?}` - `attributed=false` when no eligible transition (feedback dropped, never invented). **API key required.**
- `POST /api/v1/teaching/demonstration` - Arm a session from a constrained `instruction` and/or inject a `gesture`; `mode` = `demonstrate` | `practice`. Returns `{session_id?, trigger_gesture?, desired_action?, executed_action?, executed_action_index?}`. **API key required.**
- `GET /api/v1/teaching/qvalues` - Current policy Q-values for the encoder state, as a `{action_name: float}` map

## Configuration validation

- `GET /api/v1/config/schema` - Configuration JSON Schema
- `POST /api/v1/config/validate` - Validate proposed configuration

The validator checks a candidate configuration (YAML or env-style) against
the `AppSettings` schema without applying it. The JSON Schema endpoint
exposes the full Pydantic schema for tooling and the browser config
validator at `/config`.

## Calibration

- `GET /api/v1/calibration/servos` - List all servos
- `POST /api/v1/calibration/servos/{name}/move` - Move a servo to an angle
- `POST /api/v1/calibration/servos/{name}/release` - Release a servo
- `POST /api/v1/calibration/servos/release_all` - Release all servos
- `POST /api/v1/calibration/servos/calibrate/{name}` - Run servo calibration sequence
- `GET /api/v1/calibration/display` - Get display configuration
- `POST /api/v1/calibration/display/test_pattern` - Show a test pattern
- `POST /api/v1/calibration/display/clear` - Clear the display

## Settings (hardware test)

The settings router backs the hardware test page at `/settings/`.

### Info

- `GET /api/v1/settings/info` - Hardware & subsystem overview

### Camera

- `GET /api/v1/settings/camera/info` - Camera info
- `GET /api/v1/settings/camera/frame` - Capture a single frame
- `GET /api/v1/settings/camera/stream` - Live MJPEG camera preview

### Microphone

- `GET /api/v1/settings/mic/info` - Microphone info
- `GET /api/v1/settings/mic/level` - Current microphone input level
- `POST /api/v1/settings/mic/test` - Record and play back a mic test

### Audio output

- `GET /api/v1/settings/audio/info` - Audio output info
- `GET /api/v1/settings/audio/devices` - List available audio output devices
- `GET /api/v1/settings/audio/input-devices` - List available audio input devices
- `POST /api/v1/settings/audio/switch` - Switch the active audio output device
- `POST /api/v1/settings/audio/test-device` - Play a test tone through a specific device
- `POST /api/v1/settings/audio/tone` - Play a test tone
- `POST /api/v1/settings/audio/stop` - Stop audio playback

### Sound effects

- `GET /api/v1/settings/sound-effects` - List available sound effects
- `POST /api/v1/settings/sound-effect/{name}` - Play a sound effect

### LLM / TTS test

- `POST /api/v1/settings/tts/test` - Speak a test phrase
- `POST /api/v1/settings/llm/test` - Send a test prompt to the LLM

## System

- `GET /api/v1/system/info` - System information
- `GET /api/v1/system/logs` - Recent log entries
- `DELETE /api/v1/system/logs` - Clear log buffer
- `GET /api/v1/system/bluetooth` - Bluetooth status

## Performance

- `GET /api/v1/performance` - Combined performance summary
- `GET /api/v1/performance/frames` - Frame budget stats
- `GET /api/v1/performance/servos` - Servo latency stats
- `GET /api/v1/performance/bus` - Event bus throughput

See [Performance Profiling](performance.md).

## WebSocket

Connect to:

```text
ws://HOST:8000/api/v1/ws/events
```

The server streams event envelopes containing an event type and serialized
event data.

Example:

```json
{
  "type": "StateChanged",
  "data": {
    "previous": "idle",
    "current": "curious"
  }
}
```

Sending the literal `ping` receives `pong`.

## OpenAPI

FastAPI serves its generated OpenAPI document and interactive documentation
when the API is running. The checked-in
[`../assets/openapi.yaml`](../assets/openapi.yaml) is the human-maintained
portable specification.

When changing an endpoint, update both the implementation and the checked-in
specification.
