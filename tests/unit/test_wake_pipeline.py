"""End-to-end tests for the wake-word / audio / conversation pipeline.

These tests verify the fixes for the wake-word regression:

* wake detection is gated by the canonical robot state (never during
  LISTENING / THINKING / SPEAKING);
* DeskBot cannot wake on its own TTS output (SPEAKING gates wake detection);
* the audio loop is not blocked by the STT/LLM/TTS pipeline, so the
  microphone queue keeps draining (no saturation);
* a single wake + utterance produces exactly one logical conversation turn
  (no phantom / repeated user messages);
* empty transcriptions are dropped rather than turned into user messages;
* persistence does not duplicate messages.

The real wake-word backend (OpenWakeWordChecker) is unit-tested with a
fake model at the abstraction boundary.
"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import Callable

from tests.fakes.llm import FakeLLM

from robot.ai.conversation import ConversationManager
from robot.ai.conversation_store import InMemoryStore
from robot.ai.prompts import system_prompt
from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import WakeWordDetected
from robot.interfaces.llm import Role
from robot.interfaces.microphone import AudioChunk
from robot.services.conversation_service import ConversationService
from robot.speech.stt import MockSTT
from robot.speech.wakeword import WakeWordChecker
from robot.speech.wakeword_openwakeword import OpenWakeWordChecker

# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class QueueMicrophone:
    """Microphone backed by an asyncio.Queue the test controls.

    Lets the test push audio chunks one at a time and observe how many
    the audio loop has consumed.
    """

    sample_rate = 16_000
    channels = 1
    _sample_rate = 16_000  # read by ConversationService._audio_loop

    def __init__(self) -> None:
        self._q: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        self._closed = False
        self.consumed = 0

    async def put(self, pcm: bytes, timestamp: float) -> None:
        await self._q.put(AudioChunk(pcm=pcm, sample_rate=16_000, channels=1, timestamp=timestamp))

    async def end(self) -> None:
        await self._q.put(None)

    async def stream(self):
        while not self._closed:
            chunk = await self._q.get()
            if chunk is None:
                return
            self.consumed += 1
            yield chunk

    async def close(self) -> None:
        self._closed = True
        await self._q.put(None)


class ScriptedWakeChecker:
    """Wake checker that triggers on every call after ``warmup`` chunks.

    Simulates "loud audio is constantly present" -- exactly the scenario
    the old energy detector mishandled. With state gating, these
    triggers are ignored during an active conversation.
    """

    def __init__(self, phrase: str = "hey deskbot", warmup: int = 0) -> None:
        self.phrase = phrase
        self.warmup = warmup
        self._n = 0
        self.triggers = 0

    def check(self, pcm: bytes, timestamp: float) -> WakeWordDetected | None:
        self._n += 1
        if self._n <= self.warmup:
            return None
        self.triggers += 1
        return WakeWordDetected(phrase=self.phrase, confidence=1.0)


class BlockingTTS:
    """TTS that records calls and blocks on a gate until released.

    Used to simulate a long TTS playback during which the audio loop must
    keep draining the microphone (and must NOT wake).
    """

    def __init__(self) -> None:
        self._gate = asyncio.Event()
        self._gate.set()  # not blocking by default
        self.spoken: list[str] = []
        self.speak_started = asyncio.Event()

    def block(self) -> None:
        self.speak_started.clear()
        self._gate.clear()

    def release(self) -> None:
        self._gate.set()

    async def speak(self, text: str) -> None:
        self.spoken.append(text)
        self.speak_started.set()
        await self._gate.wait()

    async def close(self) -> None:
        self._gate.set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_for(
    predicate: Callable[[], bool], timeout: float = 2.0, interval: float = 0.005
) -> None:
    """Poll *predicate* until it returns True or timeout expires."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


def _build_service(
    *,
    bus: InMemoryEventBus,
    sm: StateMachine,
    mic: QueueMicrophone,
    tts: BlockingTTS,
    wake_checker: WakeWordChecker,
    transcript: str = "hello there",
    store: InMemoryStore | None = None,
) -> ConversationService:
    llm = FakeLLM(name="fake")
    llm.register("hello there", "Hi human!")
    conversation = ConversationManager(
        llm=llm, system_prompt=system_prompt(), store=store, conversation_id="test"
    )
    svc = ConversationService(
        bus=bus,
        state_machine=sm,
        stt=MockSTT(transcript=transcript),
        tts=tts,  # type: ignore[arg-type]
        llm=llm,
        conversation=conversation,
        microphone=mic,
        wake_checker=wake_checker,
        listen_window_s=0.1,
    )
    svc.attach()
    return svc


def _chunk_bytes() -> bytes:
    """A 30ms mono 16kHz chunk of silence."""
    return struct.pack(f"<{480}h", *([0] * 480))


# ===========================================================================
# OpenWakeWordChecker semantics (real backend, fake model)
# ===========================================================================


class _FakeOWWModel:
    """Minimal stand-in for openwakeword's Model returning a fixed score."""

    def __init__(self, phrase_key: str, score: float) -> None:
        self._predictions = {phrase_key: score}
        self.calls = 0

    def predict(self, samples: object) -> dict[str, float]:
        self.calls += 1
        return dict(self._predictions)


def _make_oww_checker(score: float, phrase: str = "hey_mycroft") -> OpenWakeWordChecker:
    checker = OpenWakeWordChecker(phrase=phrase, threshold=0.5)
    checker._model = _FakeOWWModel(phrase_key=phrase, score=score)
    # Skip warmup.
    checker._warmup_chunks = 10
    return checker


def _oww_frame() -> bytes:
    """Exactly 1280 int16 samples (one openWakeWord window)."""
    return struct.pack(f"<{1280}h", *([0] * 1280))


async def test_loud_arbitrary_audio_does_not_wake() -> None:
    """Loud audio with no wake-phrase match must NOT wake DeskBot.

    Regression for the energy detector: loudness alone is not a wake word.
    """
    # Model reports a low score (below threshold) regardless of audio.
    checker = _make_oww_checker(score=0.1)
    loud = struct.pack(f"<{1280}h", *([16000] * 1280))  # loud audio
    for t in (0.0, 0.08, 0.16, 0.24):
        result = checker.check(loud, t)
        assert result is None, "loud audio without a wake phrase must not wake"
    assert checker._model is not None
    assert checker._model.calls == 4  # type: ignore[attr-defined]


async def test_quiet_actual_wake_phrase_wakes() -> None:
    """A genuine model match wakes DeskBot even when the audio is silent.

    This proves the wake comes from the model, not from audio energy.
    """
    checker = _make_oww_checker(score=0.9)
    quiet = _oww_frame()  # all zeros -- silent
    result = checker.check(quiet, 0.08)
    assert result is not None
    assert result.phrase == "hey_mycroft"
    assert result.confidence >= 0.5


async def test_normal_speech_without_wake_phrase_does_not_wake() -> None:
    """Mid-level audio with no model match must not wake."""
    checker = _make_oww_checker(score=0.2)  # below threshold
    mid = struct.pack(f"<{1280}h", *([3000] * 1280))
    for t in (0.0, 0.08, 0.16):
        assert checker.check(mid, t) is None


# ===========================================================================
# State gating + TTS self-trigger (ConversationService audio loop)
# ===========================================================================


async def test_wake_ignored_while_speaking() -> None:
    """Wake events during SPEAKING (TTS playback) must not start a new
    conversation. This is the TTS self-trigger regression."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    sm._state = RobotState.IDLE
    mic = QueueMicrophone()
    tts = BlockingTTS()
    tts.block()  # TTS will block when first reached
    checker = ScriptedWakeChecker(warmup=0)
    svc = _build_service(bus=bus, sm=sm, mic=mic, tts=tts, wake_checker=checker)
    svc.start_audio_loop()

    # Wake + recording window.
    await mic.put(_chunk_bytes(), 0.0)  # triggers wake (state IDLE)
    await mic.put(_chunk_bytes(), 0.05)  # buffer
    await mic.put(_chunk_bytes(), 0.1)  # recording completes -> dispatch

    # Wait until TTS starts speaking (pipeline reached SPEAKING).
    await _wait_for(tts.speak_started.is_set)
    await _wait_for(lambda: sm.state is RobotState.SPEAKING)
    assert sm.state is RobotState.SPEAKING

    triggers_during_speaking_before = checker.triggers

    # Feed "loud" chunks while speaking -- the scripted checker WOULD trigger,
    # but state gating must skip wake detection entirely.
    for t in (0.2, 0.25, 0.3):
        await mic.put(_chunk_bytes(), t)
        await asyncio.sleep(0.005)
    await _wait_for(lambda: mic.consumed >= 6)

    # No new wake checks happened during SPEAKING (detection skipped).
    assert checker.triggers == triggers_during_speaking_before
    # Still only one conversation turn.
    assert sum(1 for m in svc.conversation.current.messages if m.role is Role.USER) == 1
    assert sm.state is RobotState.SPEAKING

    # Release TTS -> returns to IDLE.
    tts.release()
    await _wait_for(lambda: sm.state is RobotState.IDLE)

    # Now wake detection resumes: a new chunk triggers a second wake.
    await mic.put(_chunk_bytes(), 0.4)
    await _wait_for(lambda: sm.state is RobotState.LISTENING)
    assert checker.triggers > triggers_during_speaking_before

    svc.stop_audio_loop()
    svc.detach()


async def test_wake_ignored_while_listening_and_thinking() -> None:
    """External wake events during LISTENING/THINKING are ignored."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    sm._state = RobotState.LISTENING
    mic = QueueMicrophone()
    tts = BlockingTTS()
    svc = _build_service(bus=bus, sm=sm, mic=mic, tts=tts, wake_checker=ScriptedWakeChecker())
    svc.start_audio_loop()

    # Already LISTENING -> an external wake event must be ignored.
    await bus.publish(WakeWordDetected(phrase="hey deskbot"))
    assert sm.state is RobotState.LISTENING

    # Transition to THINKING manually and publish wake -> ignored.
    await sm.transition(RobotState.THINKING)
    await bus.publish(WakeWordDetected(phrase="hey deskbot"))
    assert sm.state is RobotState.THINKING  # type: ignore[comparison-overlap]

    svc.stop_audio_loop()  # type: ignore[unreachable]
    svc.detach()


async def test_audio_loop_not_blocked_by_tts() -> None:
    """The audio loop keeps draining the microphone while the STT/LLM/TTS
    pipeline runs. This is the microphone-queue saturation fix."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    sm._state = RobotState.IDLE
    mic = QueueMicrophone()
    tts = BlockingTTS()
    tts.block()
    svc = _build_service(bus=bus, sm=sm, mic=mic, tts=tts, wake_checker=ScriptedWakeChecker())
    svc.start_audio_loop()

    # Wake + recording window -> dispatch.
    await mic.put(_chunk_bytes(), 0.0)
    await mic.put(_chunk_bytes(), 0.05)
    await mic.put(_chunk_bytes(), 0.1)
    await _wait_for(tts.speak_started.is_set)

    # TTS is blocking. Push several more chunks -- the audio loop must
    # consume them (state gating drains & discards) instead of stalling.
    for t in (0.2, 0.25, 0.3, 0.35, 0.4):
        await mic.put(_chunk_bytes(), t)
    await _wait_for(lambda: mic.consumed >= 8)

    # The loop consumed chunks beyond the recording window while TTS blocked.
    assert mic.consumed >= 8
    assert tts.speak_started.is_set()
    # Only one turn was spoken so far.
    assert len(tts.spoken) == 1

    tts.release()
    await _wait_for(lambda: sm.state is RobotState.IDLE)
    svc.stop_audio_loop()
    svc.detach()


# ===========================================================================
# Conversation correctness
# ===========================================================================


async def test_one_wake_produces_one_user_message() -> None:
    """A single wake + utterance produces exactly one user message and one
    assistant reply -- no phantom/repeated turns."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    sm._state = RobotState.IDLE
    mic = QueueMicrophone()
    tts = BlockingTTS()
    tts.block()  # hold SPEAKING so we can feed "noise" during playback
    store = InMemoryStore()
    svc = _build_service(
        bus=bus,
        sm=sm,
        mic=mic,
        tts=tts,
        wake_checker=ScriptedWakeChecker(),
        transcript="hello there",
        store=store,
    )
    svc.start_audio_loop()

    await mic.put(_chunk_bytes(), 0.0)
    await mic.put(_chunk_bytes(), 0.05)
    await mic.put(_chunk_bytes(), 0.1)
    await _wait_for(tts.speak_started.is_set)
    await _wait_for(lambda: sm.state is RobotState.SPEAKING)

    # Feed "noise" chunks during SPEAKING that the old energy detector would
    # have treated as additional wake words. State gating must ignore them.
    for t in (0.2, 0.25, 0.3, 0.35):
        await mic.put(_chunk_bytes(), t)
    await _wait_for(lambda: mic.consumed >= 7)

    # Still exactly one turn while TTS is held in SPEAKING.
    msgs = list(svc.conversation.current.messages)
    user_msgs = [m for m in msgs if m.role is Role.USER]
    asst_msgs = [m for m in msgs if m.role is Role.ASSISTANT]
    assert len(user_msgs) == 1
    assert user_msgs[0].content == "hello there"
    assert len(asst_msgs) == 1

    tts.release()
    await _wait_for(lambda: sm.state is RobotState.IDLE)

    # Persistence: the store reflects exactly one user + one assistant.
    persisted = await store.load("test")
    assert persisted is not None
    roles = [r for r, _ in persisted]
    assert roles.count("user") == 1
    assert roles.count("assistant") == 1
    assert len(persisted) == 2

    svc.stop_audio_loop()
    svc.detach()


async def test_empty_transcript_is_dropped() -> None:
    """An empty/garbage transcription must not become a user message."""
    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    sm._state = RobotState.IDLE
    mic = QueueMicrophone()
    tts = BlockingTTS()
    svc = _build_service(
        bus=bus,
        sm=sm,
        mic=mic,
        tts=tts,
        wake_checker=ScriptedWakeChecker(),
        transcript="   ",  # whitespace -> empty after strip
    )
    svc.start_audio_loop()

    await mic.put(_chunk_bytes(), 0.0)
    await mic.put(_chunk_bytes(), 0.05)
    await mic.put(_chunk_bytes(), 0.1)

    # Empty transcript -> no SpeechRecognized, returns to IDLE, no user msg.
    await _wait_for(lambda: sm.state is RobotState.IDLE, timeout=1.5)
    assert sm.state is RobotState.IDLE
    assert sum(1 for m in svc.conversation.current.messages if m.role is Role.USER) == 0
    assert tts.spoken == []

    svc.stop_audio_loop()
    svc.detach()


async def test_conversation_history_remains_bounded() -> None:
    """Multiple turns keep the in-memory history bounded (MAX_HISTORY)."""
    from robot.ai.conversation import MAX_HISTORY

    bus = InMemoryEventBus()
    sm = StateMachine(bus=bus)
    sm._state = RobotState.IDLE
    mic = QueueMicrophone()
    tts = BlockingTTS()  # non-blocking: each turn completes fast
    svc = _build_service(
        bus=bus,
        sm=sm,
        mic=mic,
        tts=tts,
        wake_checker=ScriptedWakeChecker(),
    )
    svc.start_audio_loop()

    n_turns = MAX_HISTORY + 4
    for turn in range(n_turns):
        base = turn * 0.5
        # Only push this turn's chunks once the robot is idle, so the wake
        # is not swallowed by state gating during the previous turn.
        await _wait_for(lambda: sm.state is RobotState.IDLE)
        await mic.put(_chunk_bytes(), base + 0.0)
        await mic.put(_chunk_bytes(), base + 0.05)
        await mic.put(_chunk_bytes(), base + 0.1)
        # Wait for this turn's reply to be spoken.
        await _wait_for(lambda t=turn: len(tts.spoken) == t + 1)  # type: ignore[misc]

    await _wait_for(lambda: sm.state is RobotState.IDLE)

    msgs = list(svc.conversation.current.messages)
    assert len(msgs) <= MAX_HISTORY
    # (user, assistant) pairs bounded exactly at MAX_HISTORY.
    assert len(msgs) == MAX_HISTORY
    assert len(tts.spoken) == n_turns

    svc.stop_audio_loop()
    svc.detach()
