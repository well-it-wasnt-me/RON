"""Speech-output regression tests for the conversation pipeline.

These tests verify the contract:

    LLM response -> TTS.speak() -> AudioBuffer -> AudioOutput.play() -> speaker

The conversation service is the single orchestrator of playback: TTS
synthesises and returns an :class:`AudioBuffer`; the service then passes
that buffer to the configured :class:`AudioOutput`.  TTS backends must
**not** play internally.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from tests.fakes.llm import FakeLLM

from robot.ai.conversation import ConversationManager
from robot.ai.prompts import system_prompt
from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import SpeechRecognized
from robot.interfaces.audio import AudioBuffer
from robot.services.conversation_service import ConversationService
from robot.speech.stt import MockSTT
from robot.speech.tts import TextToSpeech


class ImmediateAudioOutput:
    """Audio output that records buffers and returns immediately."""

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


class BlockingAudioOutput:
    """Audio output that blocks until released, for testing SPEAKING state."""

    def __init__(self) -> None:
        self.played: list[AudioBuffer] = []
        self.play_started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def sample_rate(self) -> int:
        return 48_000

    @property
    def channels(self) -> int:
        return 1

    async def play(self, buffer: AudioBuffer) -> None:
        self.played.append(buffer)
        self.play_started.set()
        await self.release.wait()

    async def stop(self) -> None:
        self.release.set()

    async def close(self) -> None:
        self.release.set()


@dataclass
class SynthesizingTTS:
    """TTS that only synthesises audio; never plays it internally."""

    fail: bool = False
    spoken: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.spoken = []

    async def speak(self, text: str) -> AudioBuffer:
        self.spoken.append(text)
        if self.fail:
            raise RuntimeError("tts failed")
        return AudioBuffer(
            pcm=b"\x01\x02\x03\x04",
            sample_rate=24_000,
            channels=1,
        )

    async def close(self) -> None:
        return None


def _build_service(
    tts: TextToSpeech,
    *,
    audio: object | None = None,
) -> ConversationService:
    bus = InMemoryEventBus()
    state_machine = StateMachine(bus=bus)
    state_machine._state = RobotState.LISTENING
    llm = FakeLLM(name="fake")
    llm.register("hello there", "Hi human!")
    conversation = ConversationManager(llm=llm, system_prompt=system_prompt())
    service = ConversationService(
        bus=bus,
        state_machine=state_machine,
        stt=MockSTT(transcript="hello there"),
        tts=tts,
        llm=llm,
        conversation=conversation,
        audio=audio,  # type: ignore[arg-type]
    )
    service.attach()
    return service


@pytest.mark.asyncio
async def test_reply_reaches_tts_and_audio_output() -> None:
    """LLM response -> TTS -> AudioBuffer -> AudioOutput.play()."""
    audio = ImmediateAudioOutput()
    tts = SynthesizingTTS()
    service = _build_service(tts, audio=audio)

    await service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))

    assert tts.spoken == ["Hi human!"]
    assert len(audio.played) == 1
    assert audio.played[0].sample_rate == 24_000
    assert audio.played[0].channels == 1
    assert audio.played[0].pcm == b"\x01\x02\x03\x04"
    assert service.state_machine.state is RobotState.IDLE
    service.detach()


@pytest.mark.asyncio
async def test_speaking_remains_active_until_playback_completes() -> None:
    """SPEAKING must not transition to IDLE before playback finishes."""
    audio = BlockingAudioOutput()
    tts = SynthesizingTTS()
    service = _build_service(tts, audio=audio)

    task = asyncio.create_task(
        service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))
    )

    await asyncio.wait_for(audio.play_started.wait(), timeout=1.0)
    # Playback is blocked (release not set) - state must be SPEAKING.
    assert service.state_machine.state is RobotState.SPEAKING

    audio.release.set()
    await task

    assert service.state_machine.state is RobotState.IDLE  # type: ignore[comparison-overlap]
    service.detach()  # type: ignore[unreachable]


@pytest.mark.asyncio
async def test_tts_failure_returns_state_to_idle() -> None:
    """TTS synthesis failure is logged and state returns to IDLE."""
    service = _build_service(SynthesizingTTS(fail=True), audio=ImmediateAudioOutput())

    await service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))

    assert service.state_machine.state is RobotState.IDLE
    assert service.conversation.current.messages[-1].content == "Hi human!"
    service.detach()


@pytest.mark.asyncio
async def test_audio_playback_failure_returns_state_to_idle() -> None:
    """Audio playback failure is logged and state returns to IDLE."""
    audio = ImmediateAudioOutput(fail=True)
    service = _build_service(SynthesizingTTS(), audio=audio)

    await service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))

    assert service.state_machine.state is RobotState.IDLE
    assert len(audio.played) == 1
    service.detach()


@pytest.mark.asyncio
async def test_no_audio_output_logs_warning() -> None:
    """When no AudioOutput is configured, a warning is logged but no crash."""
    tts = SynthesizingTTS()
    service = _build_service(tts, audio=None)

    await service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))

    assert service.state_machine.state is RobotState.IDLE
    assert tts.spoken == ["Hi human!"]
    service.detach()


@pytest.mark.asyncio
async def test_final_response_passed_to_tts_exactly_once() -> None:
    """The LLM response text is passed to TTS exactly once."""
    audio = ImmediateAudioOutput()
    tts = SynthesizingTTS()
    service = _build_service(tts, audio=audio)

    await service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))

    assert tts.spoken == ["Hi human!"]
    assert len(audio.played) == 1
    service.detach()


@pytest.mark.asyncio
async def test_buffer_metadata_preserved_through_playback() -> None:
    """The AudioBuffer's sample rate and channels are preserved to the output."""
    audio = ImmediateAudioOutput()
    tts = SynthesizingTTS()
    service = _build_service(tts, audio=audio)

    await service._on_speech(SpeechRecognized(text="hello there", confidence=0.9))

    assert len(audio.played) == 1
    buf = audio.played[0]
    assert buf.sample_rate == 24_000
    assert buf.channels == 1
    assert buf.pcm == b"\x01\x02\x03\x04"
    service.detach()
