"""Tests for the speech layer."""

from __future__ import annotations

from tests.fakes.audio import FakeAudioOutput
from tests.fakes.clock import FakeClock
from tests.fakes.microphone import FakeMicrophone

from robot.events.bus import InMemoryEventBus
from robot.events.events import WakeWordDetected
from robot.speech.stt import MockSTT
from robot.speech.tts import MockTTS
from robot.speech.wakeword import MockWakeWordDetector


async def test_mock_tts_records_text() -> None:
    tts = MockTTS()
    await tts.speak("hello")
    await tts.speak("world")
    assert tts.spoken == ["hello", "world"]


async def test_mock_tts_does_not_play_audio() -> None:
    """MockTTS records text but never plays through an AudioOutput."""
    from robot.interfaces.audio import AudioBuffer

    audio = FakeAudioOutput()
    tts = MockTTS(audio=audio)
    result = await tts.speak("ping")
    assert isinstance(result, AudioBuffer)
    assert tts.spoken == ["ping"]
    # MockTTS must NOT play through audio - it produces no physical speech.
    assert len(audio.played) == 0


async def test_mock_stt_returns_text() -> None:
    stt = MockSTT(transcript="the quick brown fox")
    from robot.interfaces.microphone import AudioChunk

    result = await stt.transcribe(
        AudioChunk(pcm=b"", sample_rate=16_000, channels=1, timestamp=0.0)
    )
    assert result == "the quick brown fox"
    assert stt.calls == 1


async def test_mock_wakeword_emits_event() -> None:
    bus = InMemoryEventBus()
    mic = FakeMicrophone()
    clock = FakeClock()
    seen: list[object] = []
    bus.subscribe(WakeWordDetected, seen.append)
    detector = MockWakeWordDetector(bus=bus, microphone=mic, clock=clock, phrase="hey", delay_s=0.0)
    gen = detector.listen()
    event = await gen.__anext__()
    assert isinstance(event, WakeWordDetected)
    assert event.phrase == "hey"
    assert any(isinstance(e, WakeWordDetected) for e in seen)
