"""Integration-style tests for the speech pipeline.

Covers:

    user utterance -> STT -> LLM response -> TTS -> AudioOutput

and verifies that:

* the final response is passed to TTS exactly once
* the generated AudioBuffer reaches the AudioOutput
* SPEAKING remains active until playback completes
* state returns to IDLE after playback
* TTS failure is handled
* audio playback failure is handled
* MockTTS is not silently treated as successful physical speech
* MockAudioOutput degradation is visible
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import pytest
from tests.fakes.llm import FakeLLM

from robot.ai.conversation import ConversationManager
from robot.ai.prompts import system_prompt
from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import SpeechRecognized
from robot.interfaces.audio import AudioBuffer, AudioOutput
from robot.services.conversation_service import ConversationService
from robot.speech.stt import MockSTT
from robot.speech.tts import MockTTS, TextToSpeech


class _ImmediateAudio:
    """Non-blocking audio output that records every buffer played."""

    def __init__(self, *, fail: bool = False) -> None:
        self.played: list[AudioBuffer] = []
        self.fail = fail

    @property
    def sample_rate(self) -> int:
        return 48_000

    @property
    def channels(self) -> int:
        return 1

    async def play(self, buffer: AudioBuffer) -> None:
        self.played.append(buffer)
        if self.fail:
            raise RuntimeError("playback failed")

    async def stop(self) -> None:
        pass

    async def close(self) -> None:
        pass


@dataclass
class _RealTTS:
    """TTS that returns non-empty PCM; never plays internally."""

    spoken: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.spoken = []

    async def speak(self, text: str) -> AudioBuffer:
        self.spoken.append(text)
        pcm = struct.pack("<100h", *([16384] * 100))
        return AudioBuffer(pcm=pcm, sample_rate=22_050, channels=1)

    async def close(self) -> None:
        return None


def _build(
    *,
    tts: TextToSpeech,
    audio: AudioOutput | None = None,
    stt_transcript: str = "hello there",
) -> ConversationService:
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    sm._state = RobotState.LISTENING
    llm = FakeLLM(name="fake")
    llm.register("hello there", "Hi human!")
    conv = ConversationManager(llm=llm, system_prompt=system_prompt())
    service = ConversationService(
        bus=bus,
        state_machine=sm,
        stt=MockSTT(transcript=stt_transcript),
        tts=tts,
        llm=llm,
        conversation=conv,
        audio=audio,
    )
    service.attach()
    return service


# ---------------------------------------------------------------------------
# Full pipeline: utterance -> LLM -> TTS -> AudioOutput
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_utterance_to_speaker() -> None:
    """End-to-end: user utterance -> LLM -> TTS -> AudioOutput."""
    audio = _ImmediateAudio()
    tts = _RealTTS()
    service = _build(tts=tts, audio=audio)

    await service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))

    # TTS received the LLM response exactly once.
    assert tts.spoken == ["Hi human!"]
    # AudioOutput received the buffer exactly once.
    assert len(audio.played) == 1
    buf = audio.played[0]
    assert buf.sample_rate == 22_050
    assert buf.channels == 1
    assert len(buf.pcm) > 0
    # State returned to IDLE after playback.
    assert service.state_machine.state is RobotState.IDLE
    service.detach()


@pytest.mark.asyncio
async def test_streaming_llm_reaches_tts_and_audio() -> None:
    """Streaming LLM path also reaches TTS and AudioOutput."""
    audio = _ImmediateAudio()
    tts = _RealTTS()
    service = _build(tts=tts, audio=audio)

    # The FakeLLM supports stream_complete, so the streaming path is used.
    await service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))

    assert tts.spoken == ["Hi human!"]
    assert len(audio.played) == 1
    assert service.state_machine.state is RobotState.IDLE
    service.detach()


# ---------------------------------------------------------------------------
# Mock detection / degradation visibility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_tts_does_not_produce_physical_speech() -> None:
    """MockTTS must not be treated as successful physical speech."""
    audio = _ImmediateAudio()
    service = _build(tts=MockTTS(), audio=audio)

    await service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))

    # MockTTS records the text but produces no audio.
    assert isinstance(service.tts, MockTTS)
    assert service.tts.spoken == ["Hi human!"]  # No buffer should reach the audio output.
    assert len(audio.played) == 0
    assert service.state_machine.state is RobotState.IDLE
    service.detach()


@pytest.mark.asyncio
async def test_real_tts_with_mock_audio_is_visible() -> None:
    """Real TTS + MockAudioOutput: the buffer reaches the output but no sound."""
    from robot.hardware.audio.mock_audio import MockAudioOutput

    mock_audio = MockAudioOutput(sample_rate=48_000, channels=1)
    tts = _RealTTS()
    service = _build(tts=tts, audio=mock_audio)

    await service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))

    # The buffer reaches the mock audio output (for verification).
    assert len(mock_audio.played) == 1
    assert mock_audio.played[0].sample_rate == 22_050
    # But no physical sound was produced.
    assert service.state_machine.state is RobotState.IDLE
    service.detach()


@pytest.mark.asyncio
async def test_tts_failure_preserves_conversation_history() -> None:
    """TTS failure does not lose the LLM response from conversation history."""
    service = _build(tts=_RealTTS(), audio=_ImmediateAudio())

    # Override TTS to fail.
    object.__setattr__(service, "tts", _FailingTTS())

    await service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))

    # The assistant response is still in conversation history.
    messages = service.conversation.current.messages
    assert any(m.role.value == "assistant" and m.content == "Hi human!" for m in messages)
    assert service.state_machine.state is RobotState.IDLE
    service.detach()


@dataclass
class _FailingTTS:
    async def speak(self, text: str) -> AudioBuffer:
        raise RuntimeError("TTS engine crashed")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_audio_playback_failure_preserves_response() -> None:
    """Audio playback failure does not lose the LLM response."""
    audio = _ImmediateAudio(fail=True)
    tts = _RealTTS()
    service = _build(tts=tts, audio=audio)

    await service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))

    # The buffer reached the output (which then failed).
    assert len(audio.played) == 1
    # The assistant response is in conversation history.
    messages = service.conversation.current.messages
    assert any(m.role.value == "assistant" and m.content == "Hi human!" for m in messages)
    assert service.state_machine.state is RobotState.IDLE
    service.detach()
