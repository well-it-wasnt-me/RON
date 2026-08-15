"""Tests for the FaceOrchestrator that bridges events to the FaceAnimator."""

from __future__ import annotations

import asyncio

from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    EmotionChanged,
    EmotionName,
    LLMTokenReceived,
    StateChanged,
    WakeWordDetected,
)
from robot.face.animations import SpeakingAnimation, ThinkingDotsAnimation, WakeAnimation
from robot.face.face_orchestrator import FaceOrchestrator


class _StubAnimator:
    """Captures set_emotion and animation calls."""

    def __init__(self) -> None:
        self.emotion_calls: list[tuple[str, float]] = []
        self.speaking_animation: SpeakingAnimation | None = None
        self.thinking_animation: ThinkingDotsAnimation | None = None
        self.wake_animation: WakeAnimation | None = None

    def set_emotion(self, name: str, intensity: float = 1.0) -> None:
        self.emotion_calls.append((name, intensity))

    def set_speaking_animation(self, animation: SpeakingAnimation | None) -> None:
        self.speaking_animation = animation

    def set_thinking_animation(self, animation: ThinkingDotsAnimation | None) -> None:
        self.thinking_animation = animation

    def set_wake_animation(self, animation: WakeAnimation | None) -> None:
        self.wake_animation = animation


def _stub_emotions() -> object:
    """A stand-in EmotionEngine - only ``build()`` is needed by the orchestrator."""

    class _StubEmotions:
        def build(self, name: str) -> object:
            return object()

    return _StubEmotions()


def test_orchestrator_translates_emotion_changed() -> None:
    bus = InMemoryEventBus()
    animator = _StubAnimator()
    orchestrator = FaceOrchestrator(
        bus=bus,
        face_animator=animator,  # type: ignore[arg-type]
        emotions=_stub_emotions(),  # type: ignore[arg-type]
    )
    orchestrator.attach()
    try:

        async def publish() -> None:
            await bus.publish(
                EmotionChanged(
                    previous=EmotionName.NEUTRAL, current=EmotionName.HAPPY, intensity=0.7
                )
            )

        asyncio.run(publish())
        assert ("happy", 0.7) in animator.emotion_calls
    finally:
        orchestrator.detach()


def test_orchestrator_translates_state_changed() -> None:
    from robot.behavior.state_machine import RobotState

    bus = InMemoryEventBus()
    animator = _StubAnimator()
    orchestrator = FaceOrchestrator(
        bus=bus,
        face_animator=animator,  # type: ignore[arg-type]
        emotions=_stub_emotions(),  # type: ignore[arg-type]
    )
    orchestrator.attach()
    try:

        async def publish() -> None:
            await bus.publish(StateChanged(previous=RobotState.BOOT, current=RobotState.LISTENING))

        asyncio.run(publish())
        # LISTENING -> curious emotion.
        assert ("curious", 1.0) in animator.emotion_calls
    finally:
        orchestrator.detach()


def test_orchestrator_unknown_state_falls_back_to_neutral() -> None:
    from robot.behavior.state_machine import RobotState

    bus = InMemoryEventBus()
    animator = _StubAnimator()
    orchestrator = FaceOrchestrator(
        bus=bus,
        face_animator=animator,  # type: ignore[arg-type]
        emotions=_stub_emotions(),  # type: ignore[arg-type]
    )
    orchestrator.attach()
    try:

        async def publish() -> None:
            await bus.publish(StateChanged(previous=RobotState.BOOT, current=RobotState.ERROR))

        asyncio.run(publish())
        assert len(animator.emotion_calls) == 1
        assert animator.emotion_calls[0][0] in {"angry", "neutral"}
    finally:
        orchestrator.detach()


def test_orchestrator_first_llm_token_creates_thinking_animation() -> None:
    """First LLMTokenReceived should create a ThinkingDotsAnimation."""
    bus = InMemoryEventBus()
    animator = _StubAnimator()
    orchestrator = FaceOrchestrator(
        bus=bus,
        face_animator=animator,  # type: ignore[arg-type]
        emotions=_stub_emotions(),  # type: ignore[arg-type]
    )
    orchestrator.attach()
    try:

        async def publish() -> None:
            await bus.publish(LLMTokenReceived(token="Hello", done=False))

        asyncio.run(publish())
        assert ("thinking", 0.8) in animator.emotion_calls
        assert animator.thinking_animation is not None
        assert isinstance(animator.thinking_animation, ThinkingDotsAnimation)
    finally:
        orchestrator.detach()


def test_orchestrator_llm_done_clears_thinking_and_sets_happy() -> None:
    """LLMTokenReceived with done=True should clear thinking animation and set happy."""
    bus = InMemoryEventBus()
    animator = _StubAnimator()
    orchestrator = FaceOrchestrator(
        bus=bus,
        face_animator=animator,  # type: ignore[arg-type]
        emotions=_stub_emotions(),  # type: ignore[arg-type]
    )
    orchestrator.attach()
    try:

        async def publish() -> None:
            # First token: thinking
            await bus.publish(LLMTokenReceived(token="Hello", done=False))
            assert animator.thinking_animation is not None
            # Done token: happy, clear thinking
            await bus.publish(LLMTokenReceived(token="", done=True))

        asyncio.run(publish())
        assert ("thinking", 0.8) in animator.emotion_calls
        assert ("happy", 1.0) in animator.emotion_calls
        assert animator.thinking_animation is None
    finally:
        orchestrator.detach()


def test_orchestrator_llm_streaming_resets() -> None:
    """After done=True, a new token starts a new thinking session."""
    bus = InMemoryEventBus()
    animator = _StubAnimator()
    orchestrator = FaceOrchestrator(
        bus=bus,
        face_animator=animator,  # type: ignore[arg-type]
        emotions=_stub_emotions(),  # type: ignore[arg-type]
    )
    orchestrator.attach()
    try:

        async def publish() -> None:
            # First stream
            await bus.publish(LLMTokenReceived(token="Hello", done=False))
            await bus.publish(LLMTokenReceived(token=" world", done=True))
            # Second stream
            await bus.publish(LLMTokenReceived(token="New", done=False))

        asyncio.run(publish())
        # Should have two thinking entries
        thinking_calls = [c for c in animator.emotion_calls if c[0] == "thinking"]
        assert len(thinking_calls) == 2
    finally:
        orchestrator.detach()


def test_orchestrator_llm_accumulates_reply_text() -> None:
    """Tokens should be accumulated into _reply_text for speaking animation."""
    bus = InMemoryEventBus()
    animator = _StubAnimator()
    orchestrator = FaceOrchestrator(
        bus=bus,
        face_animator=animator,  # type: ignore[arg-type]
        emotions=_stub_emotions(),  # type: ignore[arg-type]
    )
    orchestrator.attach()
    try:

        async def publish() -> None:
            await bus.publish(LLMTokenReceived(token="Hello", done=False))
            await bus.publish(LLMTokenReceived(token=" world", done=False))
            await bus.publish(LLMTokenReceived(token="!", done=True))

        asyncio.run(publish())
        assert orchestrator._reply_text == "Hello world!"
    finally:
        orchestrator.detach()


def test_orchestrator_speaking_state_creates_speaking_animation() -> None:
    """Transitioning to SPEAKING should create a SpeakingAnimation."""
    from robot.behavior.state_machine import RobotState

    bus = InMemoryEventBus()
    animator = _StubAnimator()
    orchestrator = FaceOrchestrator(
        bus=bus,
        face_animator=animator,  # type: ignore[arg-type]
        emotions=_stub_emotions(),  # type: ignore[arg-type]
    )
    orchestrator.attach()
    # Pre-populate reply text by simulating LLM tokens.
    orchestrator._streaming = True
    orchestrator._reply_text = "Hello there!"
    try:

        async def publish() -> None:
            await bus.publish(
                StateChanged(previous=RobotState.THINKING, current=RobotState.SPEAKING)
            )

        asyncio.run(publish())
        assert animator.speaking_animation is not None
        assert isinstance(animator.speaking_animation, SpeakingAnimation)
    finally:
        orchestrator.detach()


def test_orchestrator_idle_state_clears_speaking_animation() -> None:
    """Transitioning to IDLE should clear the speaking animation."""
    from robot.behavior.state_machine import RobotState

    bus = InMemoryEventBus()
    animator = _StubAnimator()
    orchestrator = FaceOrchestrator(
        bus=bus,
        face_animator=animator,  # type: ignore[arg-type]
        emotions=_stub_emotions(),  # type: ignore[arg-type]
    )
    orchestrator.attach()
    # Set a speaking animation.
    animator.speaking_animation = SpeakingAnimation(text="test")
    orchestrator._reply_text = "test"
    try:

        async def publish() -> None:
            await bus.publish(StateChanged(previous=RobotState.SPEAKING, current=RobotState.IDLE))

        asyncio.run(publish())
        assert animator.speaking_animation is None
        assert orchestrator._reply_text == ""  # type: ignore[unreachable]
    finally:
        orchestrator.detach()


def test_orchestrator_wake_word_creates_wake_animation() -> None:
    """WakeWordDetected should create a WakeAnimation."""
    bus = InMemoryEventBus()
    animator = _StubAnimator()
    orchestrator = FaceOrchestrator(
        bus=bus,
        face_animator=animator,  # type: ignore[arg-type]
        emotions=_stub_emotions(),  # type: ignore[arg-type]
    )
    orchestrator.attach()
    try:

        async def publish() -> None:
            await bus.publish(WakeWordDetected(phrase="hey deskbot", confidence=0.95))

        asyncio.run(publish())
        assert animator.wake_animation is not None
        assert isinstance(animator.wake_animation, WakeAnimation)
    finally:
        orchestrator.detach()


def test_orchestrator_reply_text_resets_on_new_stream() -> None:
    """_reply_text should be reset when a new streaming session starts."""
    bus = InMemoryEventBus()
    animator = _StubAnimator()
    orchestrator = FaceOrchestrator(
        bus=bus,
        face_animator=animator,  # type: ignore[arg-type]
        emotions=_stub_emotions(),  # type: ignore[arg-type]
    )
    orchestrator.attach()
    # Set up initial state
    orchestrator._streaming = True
    orchestrator._reply_text = "old text"
    try:

        async def publish() -> None:
            # done=True ends the stream
            await bus.publish(LLMTokenReceived(token="", done=True))

        asyncio.run(publish())
        assert orchestrator._streaming is False

        # New stream should reset _reply_text
        async def publish_new() -> None:
            await bus.publish(LLMTokenReceived(token="New", done=False))

        asyncio.run(publish_new())
        assert orchestrator._reply_text == "New"
    finally:
        orchestrator.detach()
