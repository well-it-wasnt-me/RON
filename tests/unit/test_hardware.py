"""Tests for the mock hardware."""

from __future__ import annotations

import pytest

from robot.errors import ServoError
from robot.hardware.audio.mock_audio import MockAudioOutput
from robot.hardware.displays.mock_display import MockDisplay
from robot.hardware.sensors.mock_camera import MockCamera
from robot.hardware.sensors.mock_microphone import MockMicrophone
from robot.hardware.servos.mock_servo import MockServo, MockServoBus


async def test_mock_display_pushes_frame() -> None:
    d = MockDisplay(width=16, height=16)
    from robot.interfaces.display import EyeFrame

    frame = EyeFrame(width=16, height=16, pixels=b"\xff" * (16 * 16 * 3))
    await d.show(frame)
    assert d.frames_pushed == 1
    assert d.last_frame is frame


async def test_mock_display_clear() -> None:
    d = MockDisplay(width=8, height=8)
    await d.clear()
    assert d.last_frame is not None
    assert d.last_frame.pixels == b"\x00\x00\x00" * (8 * 8)


async def test_mock_servo_records_moves() -> None:
    s = MockServo(name="pan", min_angle=-90, max_angle=90)
    await s.move_to(45.0)
    await s.move_to(-30.0)
    assert s.angle == -30.0
    assert s.history == [(45.0, 0.4), (-30.0, 0.4)]


async def test_mock_servo_rejects_invalid_angle() -> None:
    s = MockServo(name="pan", min_angle=-90, max_angle=90)
    with pytest.raises(ServoError):
        await s.move_to(180.0)


async def test_mock_servo_bus_lookup() -> None:
    bus = MockServoBus({"a": MockServo(name="a")})
    assert bus.get("a").name == "a"
    with pytest.raises(ServoError):
        bus.get("missing")


async def test_mock_audio_records_playback() -> None:
    from robot.interfaces.audio import AudioBuffer

    a = MockAudioOutput()
    buf = AudioBuffer(pcm=b"\x00\x00", sample_rate=22050, channels=1)
    await a.play(buf)
    assert len(a.played) == 1
    assert a.played[0].pcm == b"\x00\x00"
    assert a.played[0].sample_rate == 22050


async def test_mock_microphone_streams() -> None:
    m = MockMicrophone(frame_ms=5)
    stream = m.stream()
    chunk = await stream.__anext__()
    assert len(chunk.pcm) > 0
    assert chunk.sample_rate == m.sample_rate
    await m.close()


async def test_mock_camera_capture() -> None:
    c = MockCamera(width=10, height=10)
    frame = await c.capture()
    assert frame.width == 10
    assert frame.height == 10
    assert c.captured == 1
