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


def test_create_temp_mic_rtsp_returns_real_rtsp_mic() -> None:
    """An RtspMicrophone must mirror to a real RtspMicrophone, not MockMicrophone.

    Previously ``_create_temp_mic`` only handled Mock/Usb microphones and
    fell back to MockMicrophone (silence) for RtspMicrophone, while the
    /mic/stream header still advertised ``RtspMicrophone`` / ``is_mock=false``
    — so the Live View ears card reported "all ok" but pumped silence.
    """
    from robot.api.settings import _create_temp_mic
    from robot.hardware.sensors.rtsp_microphone import RtspMicrophone

    original = RtspMicrophone(
        url="rtsp://dummy:8554/live",
        output_sample_rate=16_000,
        channels=1,
        frame_ms=30,
        transport="tcp",
    )
    temp = _create_temp_mic(original, settings=None, sample_rate=16_000, channels=1, frame_ms=30)
    assert isinstance(temp, RtspMicrophone)
    assert temp.url == "rtsp://dummy:8554/live"
    assert temp.transport == "tcp"
    assert temp.output_sample_rate == 16_000
    # RtspMicrophone decodes to mono; the temp mic must be mono regardless
    # of the requested channel count so it never raises and falls back.
    assert temp.channels == 1
