"""Tests for the live microphone WebSocket stream (``/settings/mic/stream``).

The Live View panel's "hear the world as RON" ear card is backed by this
endpoint, which streams raw s16le PCM as binary frames after a text header
carrying the audio format. These tests use Starlette's ``TestClient`` (the
only way to drive a WebSocket endpoint in-process) with a mock microphone
wired into the bridge.
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from starlette.testclient import TestClient

from robot.api.app import create_app
from robot.api.state_bridge import StateBridge
from robot.config import AppSettings
from robot.hardware.sensors.mock_microphone import MockMicrophone


def _wired_app() -> FastAPI:
    """An app whose bridge has a mock microphone (silence) attached."""
    settings = AppSettings(_env_file=None, env="testing", log_level="WARNING")
    app = create_app(settings=settings)
    app.state.bridge = StateBridge(
        microphone=MockMicrophone(sample_rate=16000, channels=1, frame_ms=30)
    )
    return app


def test_mic_stream_header_and_pcm() -> None:
    """A wired mic yields a format header then raw s16le silence frames."""
    app = _wired_app()
    with TestClient(app) as client, client.websocket_connect("/api/v1/settings/mic/stream") as ws:
        header = json.loads(ws.receive_text())
        assert header["sample_rate"] == 16000
        assert header["channels"] == 1
        assert header["frame_ms"] == 30
        assert header["is_mock"] is True
        assert header["type"] == "MockMicrophone"

        pcm = ws.receive_bytes()
        # 30 ms at 16 kHz mono, s16le -> 480 samples * 2 bytes.
        assert len(pcm) == 960
        # MockMicrophone emits deterministic silence.
        assert pcm == b"\x00" * 960


def test_mic_stream_no_microphone() -> None:
    """With no mic on the bridge the endpoint sends an error and closes."""
    settings = AppSettings(_env_file=None, env="testing", log_level="WARNING")
    app = create_app(settings=settings)  # bridge has no microphone
    with TestClient(app) as client, client.websocket_connect("/api/v1/settings/mic/stream") as ws:
        msg = json.loads(ws.receive_text())
        assert msg == {"error": "no_microphone"}
