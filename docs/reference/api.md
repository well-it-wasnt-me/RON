# REST API and WebSocket API

The FastAPI application is created by `robot.api.create_app()`.

The API is mounted under `/api/v1`.

## REST endpoints

### Health

- `GET /api/v1/health`
- `GET /api/v1/version`

### State and status

- `GET /api/v1/state`
- `POST /api/v1/state`
- `GET /api/v1/config`
- `GET /api/v1/perception`
- `GET /api/v1/audio`
- `GET /api/v1/conversation`

### Commands

- `POST /api/v1/speak`
- `POST /api/v1/speak-direct`
- `POST /api/v1/emotion`

`/speak` injects text into the conversational pipeline. `/speak-direct`
bypasses STT and LLM and sends text directly to the configured TTS.

### Conversations

- `GET /api/v1/conversations`
- `GET /api/v1/conversations/{conversation_id}`
- `DELETE /api/v1/conversations/{conversation_id}`

These operate on the configured conversation store.

### Calibration

- `GET /api/v1/calibration/servos`
- `POST /api/v1/calibration/servos/{name}/move`
- `POST /api/v1/calibration/servos/{name}/release`
- `POST /api/v1/calibration/servos/release_all`
- `POST /api/v1/calibration/servos/calibrate/{name}`
- `GET /api/v1/calibration/display`
- `POST /api/v1/calibration/display/test_pattern`
- `POST /api/v1/calibration/display/clear`

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
