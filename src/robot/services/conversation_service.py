"""Glues STT, LLM, TTS, microphone, and the wake-word checker.

The :class:`ConversationService` is the single integration point
between audio I/O and the language-model pipeline:

* A :class:`WakeWordChecker` (or any object matching the protocol)
  analyses each audio chunk for wake-word triggers.
* On a wake-word detection the service transitions the robot to
  :class:`RobotState.LISTENING`, buffers ``listen_window_s`` seconds
  of audio, then publishes a :class:`SpeechRecognized` event.
* The STT pipeline consumes that event, the LLM produces a reply,
  the TTS pipeline speaks it, and the state machine returns to
* ``IDLE``.

When the LLM responds with tool calls (function calling), the
service dispatches them through the :class:`ToolExecutor` and
re-calls the LLM with the results, continuing the conversation.

When ``wake_checker`` is ``None`` the audio loop skips wake detection
entirely - the service only responds to :class:`WakeWordDetected`
events published by external components.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from robot.ai.conversation import ConversationManager
from robot.ai.llm_mock import MockLLM
from robot.ai.memory import Memory
from robot.ai.preferences import PreferenceTracker
from robot.ai.prompts import system_prompt
from robot.ai.tools.executor import ToolExecutor
from robot.ai.tools.registry import ToolRegistry
from robot.ai.vector_memory import VectorMemory
from robot.behavior.state_machine import RobotState, StateMachine
from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    BotReply,
    EmotionChanged,
    EmotionName,
    LLMTokenReceived,
    SpeechRecognized,
    StateChanged,
    WakeWordDetected,
)
from robot.interfaces.audio import AudioBuffer, AudioOutput, convert_audio
from robot.interfaces.llm import LLM, LLMResponse, Message, Role, ToolCall
from robot.interfaces.microphone import AudioChunk, Microphone
from robot.learning.feedback_service import FeedbackService
from robot.logging import get_logger
from robot.speech.stt import MockSTT, SpeechToText
from robot.speech.tts import MockTTS, TextToSpeech
from robot.speech.wakeword import NullWakeWordChecker, WakeWordChecker

if TYPE_CHECKING:
    from robot.learning.teaching_controller import TeachingController

_log = get_logger("services.conversation")

#: Utterances treated as positive human feedback ("good robot").
_POSITIVE_FEEDBACK_WORDS: frozenset[str] = frozenset(
    {"good", "yes", "nice", "right", "correct", "great", "yep", "yeah", "perfect"}
)
#: Utterances treated as negative human feedback.
_NEGATIVE_FEEDBACK_WORDS: frozenset[str] = frozenset(
    {"no", "wrong", "don't", "dont", "nope", "bad", "incorrect", "stop", "not"}
)
#: Multi-word positive phrases; matched as substrings (lowercased).
_POSITIVE_FEEDBACK_PHRASES: frozenset[str] = frozenset({"that's good", "thats good", "good job"})
#: Multi-word negative phrases.
_NEGATIVE_FEEDBACK_PHRASES: frozenset[str] = frozenset({"not that", "no don't", "no dont"})

# Maximum number of tool-call round trips before giving up and
# speaking whatever text we have. Prevents infinite loops if the
# LLM keeps calling tools.
_MAX_TOOL_ROUNDS = 5

# Robot states in which wake-word detection is allowed. Wake detection is
# deliberately gated by the canonical state machine (not a separate flag):
# it only runs when the robot is idle/awake, never while listening,
# thinking, or speaking. This prevents DeskBot's own TTS output (which
# plays during SPEAKING) from re-triggering a wake word and prevents a new
# conversation from starting while one is already active.
_WAKE_ALLOWED_STATES: frozenset[RobotState] = frozenset(
    {RobotState.IDLE, RobotState.CURIOUS, RobotState.SLEEPING}
)


@dataclass(slots=True)
class ConversationService:
    """Listens for wake-word and speech events, drives the LLM, and replies.

    When ``tool_registry`` and ``tool_executor`` are provided, the
    service will handle LLM tool calls by dispatching them through
    the executor and re-calling the LLM with the results.
    """

    bus: InMemoryEventBus
    state_machine: StateMachine
    stt: SpeechToText
    tts: TextToSpeech
    llm: LLM
    conversation: ConversationManager
    #: Optional microphone used by the audio loop. When ``None`` the
    #: service falls back to event-driven mode (no audio capture).
    microphone: Microphone | None = None
    #: Wake-word checker that analyses each audio chunk for triggers.
    #: When ``None``, a :class:`NullWakeWordChecker` is used (never triggers).
    wake_checker: WakeWordChecker | None = None
    memory: Memory | VectorMemory | None = None
    memory_recall_limit: int = 5
    #: Optional preference tracker for learning user preferences.
    preference_tracker: PreferenceTracker | None = None
    #: Tool registry for LLM function calling. When ``None``, tool
    #: calling is disabled and the LLM is called without tools.
    tool_registry: ToolRegistry | None = None
    #: Tool executor for dispatching tool calls. Required when
    #: ``tool_registry`` is set.
    tool_executor: ToolExecutor | None = None
    #: Seconds of audio to capture after a wake trigger.
    listen_window_s: float = 1.5
    #: Audio output for physical playback of TTS audio. When ``None``,
    #: TTS audio is synthesised but not played through a speaker.
    audio: AudioOutput | None = None
    #: Optional human-feedback service. When set, a recognised utterance that
    #: reads as praise/correction ("good"/"no") is attributed to the most-recent
    #: eligible transition as post-hoc feedback. The turn still goes to the
    #: LLM — feedback is a side effect, never a replacement for the reply. When
    #: ``None`` no utterance is ever treated as feedback.
    feedback_service: FeedbackService | None = None
    #: Optional teaching controller (Phase 8). When set, ``_on_speech`` first
    #: tries to parse a teaching instruction; on a match it starts a session
    #: and acknowledges instead of running a normal LLM turn. Left ``None``
    #: outside teaching mode.
    teaching_controller: TeachingController | None = None

    _audio_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _recording_buffer: bytearray = field(default_factory=bytearray, init=False)
    _recording_until_s: float = field(default=0.0, init=False)
    _recording_sr: int = field(default=16000, init=False)
    _in_listening: bool = field(default=False, init=False)
    #: Background task running the STT -> LLM -> TTS pipeline for the
    #: current utterance. Kept so it is not garbage-collected and so we
    #: can observe failures. With state-gated wake detection, at most one
    #: is active at a time.
    _speech_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _chunks_seen: int = field(default=0, init=False)
    _chunks_consumed: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        # Use a null checker when none is provided so the audio loop
        # can always call .check() without a None guard.
        if self.wake_checker is None:
            object.__setattr__(self, "wake_checker", NullWakeWordChecker())

    def attach(self) -> None:
        self.bus.subscribe(WakeWordDetected, self._on_wake_word)
        self.bus.subscribe(SpeechRecognized, self._on_speech)
        self.bus.subscribe(StateChanged, self._on_state_change)

    def detach(self) -> None:
        self.bus.unsubscribe(WakeWordDetected, self._on_wake_word)
        self.bus.unsubscribe(SpeechRecognized, self._on_speech)
        self.bus.unsubscribe(StateChanged, self._on_state_change)

    # ------------------------------------------------------------------ text input
    async def handle_user_text(self, text: str, *, source: str = "text") -> None:
        """Process a text user utterance through the normal conversation pipeline.

        This is the canonical entry point for text-based user input
        (terminal chat, API, MQTT).  It transitions to LISTENING and
        publishes :class:`SpeechRecognized`, which the existing
        `:meth:`_on_speech` handler processes through LLM -> TTS ->
        audio playback, exactly like a spoken utterance.

        Because the event bus awaits all subscribers, this method
        returns only after the full conversation turn (including TTS
        playback) has completed and the state machine is back to IDLE.

        Blank input is ignored.
        """
        if not text.strip():
            return
        _log.info("conversation.input", source=source, text_length=len(text))
        state = self.state_machine.state
        if state is not RobotState.LISTENING:
            if state not in _WAKE_ALLOWED_STATES:
                _log.warning(
                    "conversation.text_input_rejected",
                    state=state.value,
                    reason="state does not allow listening",
                )
                return
            await self.state_machine.transition(RobotState.LISTENING)
        await self.bus.publish(SpeechRecognized(text=text, confidence=1.0))

    # ------------------------------------------------------------------ audio loop
    def start_audio_loop(self) -> None:
        """Start the background task that consumes the microphone stream."""
        if self.microphone is None:
            _log.info("conversation.audio_loop.skipped", reason="no microphone")
            return
        if self._audio_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _log.warning("conversation.audio_loop.no_event_loop")
            return
        self._audio_task = loop.create_task(self._audio_loop(), name="ConversationService-audio")
        _log.info(
            "conversation.audio_loop.started",
            microphone=type(self.microphone).__name__,
            wake_checker=type(self.wake_checker).__name__,
            listen_window_s=self.listen_window_s,
        )

    def stop_audio_loop(self) -> None:
        task = self._audio_task
        if task is None:
            return
        task.cancel()
        self._audio_task = None

    async def _audio_loop(self) -> None:
        """Consume the microphone stream and run the wake-word checker.

        Wake detection is gated by the canonical robot state (see
        :data:`_WAKE_ALLOWED_STATES`): it only runs when the robot is idle
        or asleep, never while listening, thinking, or speaking. This is
        what prevents DeskBot from waking on its own TTS output (played
        during SPEAKING) and prevents a new conversation from starting
        while one is active.

        The conversation turn (STT -> LLM -> TTS) is dispatched as a
        background task (:meth:`_spawn_speech_dispatch`) so this loop keeps
        draining the microphone queue even while the pipeline runs. This
        prevents the microphone queue from saturating and dropping
        thousands of chunks during a single conversation turn.
        """
        assert self.microphone is not None
        self._recording_sr = int(getattr(self.microphone, "sample_rate", 16000))
        try:
            async for chunk in self.microphone.stream():
                self._chunks_seen += 1
                self._chunks_consumed += 1
                state = self.state_machine.state
                wake_chunk = self._prepare_chunk_for_wake(chunk)

                # Recording the user utterance after a wake event.
                if self._in_listening:
                    self._recording_buffer.extend(wake_chunk.pcm)
                    self._recording_sr = wake_chunk.sample_rate
                    if wake_chunk.timestamp >= self._recording_until_s:
                        pcm = bytes(self._recording_buffer)
                        self._recording_buffer.clear()
                        self._in_listening = False
                        # Hand off the conversation turn WITHOUT blocking
                        # the audio loop so the mic queue keeps draining.
                        self._spawn_speech_dispatch(pcm)
                    continue

                # Only run wake detection in idle/awake states. While
                # LISTENING (dispatch pending), THINKING, SPEAKING, or
                # ERROR we consume and discard chunks to keep the queue
                # drained and to avoid self-triggering on TTS output.
                if state in _WAKE_ALLOWED_STATES:
                    assert self.wake_checker is not None
                    # Run the wake check synchronously. On the Pi 5 an
                    # openWakeWord predict is ~7 ms -- cheap enough to run
                    # inline without yielding. Yielding per chunk (e.g. via
                    # ``run_in_executor``) was tried and *caused* drops: it
                    # forced the audio loop to re-schedule on every chunk,
                    # and event-loop congestion (perception / display /
                    # event subscribers) delayed each continuation by ~50
                    # ms, throttling the drain to ~16 chunks/s while the
                    # paced producer delivers 33/s -- so the queue
                    # saturated and ~50% of audio was dropped. Sync check
                    # drains at full realtime speed with zero drops. If a
                    # future wake backend ever exceeds realtime here, the
                    # fix is to decouple wake detection onto its own
                    # consumer task (not to await an executor per chunk).
                    wake_event = self.wake_checker.check(
                        wake_chunk.pcm, wake_chunk.timestamp
                    )
                    if wake_event is not None:
                        self._in_listening = True
                        self._recording_buffer.clear()
                        self._recording_buffer.extend(wake_chunk.pcm)
                        self._recording_sr = wake_chunk.sample_rate
                        self._recording_until_s = wake_chunk.timestamp + self.listen_window_s
                        _log.info(
                            "conversation.wake_detected",
                            phrase=wake_event.phrase,
                            confidence=wake_event.confidence,
                            window_s=self.listen_window_s,
                            state=state.value,
                        )
                        await self.bus.publish(wake_event)
                        await self.state_machine.transition(RobotState.LISTENING)
                elif self._chunks_seen == 1:
                    _log.warning(
                        "conversation.wake_detection_gated",
                        state=state.value,
                        allowed_states=sorted(s.value for s in _WAKE_ALLOWED_STATES),
                    )

                # Rate-limited diagnostic tick (~every 100 chunks).
                if self._chunks_seen % 100 == 0:
                    _log.debug(
                        "conversation.audio_loop.tick",
                        state=state.value,
                        in_listening=self._in_listening,
                        chunks=self._chunks_seen,
                        speech_task_running=(
                            self._speech_task is not None and not self._speech_task.done()
                        ),
                        microphone_chunks_consumed=self._chunks_consumed,
                        microphone_runtime=getattr(
                            self.microphone, "runtime_stats", lambda: None
                        )(),
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.exception("conversation.audio_loop.crashed")

    async def _transcribe(self, pcm: bytes) -> str:
        """Run STT on a PCM buffer; empty string on failure."""
        try:
            text = await self.stt.transcribe(
                AudioChunk(
                    pcm=pcm,
                    sample_rate=self._recording_sr,
                    channels=1,
                    timestamp=0.0,
                )
            )
            _log.info(
                "stt.completed",
                sample_rate=self._recording_sr,
                bytes=len(pcm),
                text_length=len(text.strip()),
            )
            return text
        except Exception:
            _log.exception("conversation.transcribe_failed")
            return ""

    def _spawn_speech_dispatch(self, pcm: bytes) -> None:
        """Dispatch the recognised utterance as a background task.

        Called by the audio loop when the listening window completes so
        the loop can keep draining the microphone queue while STT/LLM/TTS
        run. With state-gated wake detection, at most one dispatch task is
        active at a time: a new wake can only fire once the robot returns
        to an idle state and the previous turn has finished.
        """
        if self._speech_task is not None and not self._speech_task.done():
            # Should not happen with state gating; be defensive.
            _log.warning(
                "conversation.speech_dispatch.superseded",
                state=self.state_machine.state.value,
            )
            self._speech_task.cancel()
        task = asyncio.create_task(self._dispatch_speech(pcm), name="ConversationService-speech")
        self._speech_task = task
        task.add_done_callback(self._on_speech_task_done)

    @staticmethod
    def _on_speech_task_done(task: asyncio.Task[None]) -> None:
        """Log dispatch failures; swallow cancellation."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            _log.error("conversation.speech_dispatch.failed", error=str(exc))

    async def _dispatch_speech(self, pcm: bytes) -> None:
        """Transcribe the recorded utterance and publish SpeechRecognized.

        Runs as a background task so the audio loop is not blocked. The
        :meth:`_on_speech` handler (subscribed to SpeechRecognized) drives
        the LLM/TTS pipeline and the state transitions back to IDLE.

        Empty transcriptions are dropped rather than turned into user
        messages: an empty/garbage transcript must not become a
        conversation turn. This is part of the fix for phantom repeated
        turns caused by the old energy wake-word false triggers.
        """
        text = await self._transcribe(pcm)
        if not text.strip():
            _log.info(
                "conversation.empty_transcript",
                state=self.state_machine.state.value,
            )
            # Return to IDLE so wake detection can resume.
            if self.state_machine.state is RobotState.LISTENING:
                await self.state_machine.transition(RobotState.IDLE)
            return
        await self.bus.publish(SpeechRecognized(text=text, confidence=0.8))

    # ------------------------------------------------------------------ event handlers
    async def _on_wake_word(self, event: WakeWordDetected) -> None:
        state = self.state_machine.state
        if state not in _WAKE_ALLOWED_STATES:
            # Ignore wake events while listening, thinking, or speaking.
            # This blocks the self-trigger path (SPEAKING -> wake ->
            # LISTENING) and prevents a new conversation from starting
            # while one is already active.
            _log.info(
                "conversation.wake_ignored",
                phrase=event.phrase,
                state=state.value,
                reason="active conversation",
            )
            return
        _log.info("conversation.wake_word", phrase=event.phrase, state=state.value)
        await self.state_machine.transition(RobotState.LISTENING)
        await self.bus.publish(
            EmotionChanged(
                previous=EmotionName.NEUTRAL,
                current=EmotionName.CURIOUS,
                intensity=0.8,
            )
        )

    @staticmethod
    def _match_feedback(text: str) -> str | None:  # noqa: PLR0911
        """Classify an utterance as positive/negative feedback, or ``None``.

        A small, deliberately constrained matcher (no LLM): positive cues
        like ``"good"``/``"yes"``/``"that's good"`` and negative cues like
        ``"no"``/``"wrong"``/``"don't"``. Returns ``"positive"``/``"negative"``
        or ``None`` when the utterance is not feedback. Phrase matches take
        priority over single-word matches so ``"that's good"`` is positive
        rather than tripping on any other word.
        """
        lowered = text.strip().lower()
        if not lowered:
            return None
        # Multi-word phrases first (longer signal is more specific).
        for phrase in _NEGATIVE_FEEDBACK_PHRASES:
            if phrase in lowered:
                return "negative"
        for phrase in _POSITIVE_FEEDBACK_PHRASES:
            if phrase in lowered:
                return "positive"
        # Single-token utterance (or utterance whose first token is a cue).
        tokens = lowered.replace(".", "").replace(",", "").replace("!", "").split()
        if not tokens:
            return None
        first = tokens[0]
        if first in _POSITIVE_FEEDBACK_WORDS:
            return "positive"
        if first in _NEGATIVE_FEEDBACK_WORDS:
            return "negative"
        return None

    async def _on_speech(self, event: SpeechRecognized) -> None:
        if self.state_machine.state is not RobotState.LISTENING:
            return
        await self.state_machine.transition(RobotState.THINKING)

        # Teaching instructions are handled *before* the LLM turn and without
        # it: a constrained parser recognises "RON, when I wave, wave back",
        # arms a teaching session, and we acknowledge + return. The LLM never
        # decides what action the robot should learn. When no teaching
        # controller is wired, or the utterance is not an instruction, the
        # turn falls through to the normal LLM conversation below.
        if self.teaching_controller is not None:
            session_id = self.teaching_controller.arm_from_instruction(event.text)
            if session_id is not None:
                _log.info(
                    "conversation.teaching_armed",
                    session_id=session_id,
                    text=event.text,
                )
                ack = "Got it. Show me the gesture and I'll respond."
                await self.bus.publish(BotReply(text=ack, user_text=event.text))
                await self._speak_reply(ack)
                await self.state_machine.transition(RobotState.LISTENING)
                return

        # Extract preferences from the user utterance.
        if self.preference_tracker is not None:
            self.preference_tracker.process_user_text(event.text)

        # Human feedback is a *side effect*: if the utterance reads as
        # praise/correction AND a recent eligible transition exists, attribute
        # it post-hoc. The turn still proceeds to the LLM below. Without a
        # feedback service wired, no utterance is ever treated as feedback.
        if self.feedback_service is not None:
            polarity = self._match_feedback(event.text)
            if polarity is not None:
                await self.feedback_service.handle_feedback(
                    polarity=+1 if polarity == "positive" else -1,
                    source="speech",
                    text=event.text,
                )

        memory_context = self._memory_context(event.text)

        # Build tool schemas if tool calling is enabled.
        tool_schemas: list[dict[str, Any]] | None = None
        if self.tool_registry and self.tool_registry.tool_count > 0:
            tool_schemas = self.tool_registry.get_schemas()

        # Use streaming if the LLM supports it; fall back to one-shot.
        if hasattr(self.llm, "stream_complete") and callable(self.llm.stream_complete):
            await self._handle_streaming(event.text, memory_context, tool_schemas)
        else:
            await self._handle_one_shot(event.text, memory_context, tool_schemas)

    # ------------------------------------------------------------------ one-shot LLM
    async def _handle_one_shot(
        self,
        user_text: str,
        memory_context: str,
        tool_schemas: list[dict[str, Any]] | None,
    ) -> None:
        """Handle a conversation turn using a single LLM call (with optional tool loop)."""
        self.conversation.current.add_user(user_text)
        messages = self.conversation.messages_for(memory_context)

        for _ in range(_MAX_TOOL_ROUNDS):
            if hasattr(self.llm, "complete_with_tools") and callable(self.llm.complete_with_tools):
                response: LLMResponse = await self.llm.complete_with_tools(
                    messages, tools=tool_schemas
                )
            else:
                text = await self.llm.complete(messages)
                response = LLMResponse(text=text)

            if response.tool_calls:
                # Dispatch tool calls and append results to messages.
                tool_results = await self._execute_tool_calls(response.tool_calls)
                # Add the assistant's tool-call message with tool_calls attached
                # so OpenAI-compatible endpoints can correlate tool results.
                assistant_content = response.text or ""
                messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        content=assistant_content,
                        tool_calls=response.tool_calls,
                    )
                )
                for tc, result in zip(response.tool_calls, tool_results, strict=False):
                    messages.append(Message(role=Role.TOOL, content=result, tool_call_id=tc.id))
                _log.info(
                    "conversation.tool_round",
                    calls=len(response.tool_calls),
                    results=len(tool_results),
                )
                continue

            # No tool calls - we have a final text response.
            self.conversation.current.add_assistant(response.text)
            await self.conversation.save()
            self._remember_exchange(user_text, response.text)

            _log.info("conversation.reply", text=response.text)
            await self.bus.publish(BotReply(text=response.text, user_text=user_text))
            await self._speak_reply(response.text)
            return

        # If we exhausted the tool round limit, speak whatever text we have.
        final_reply = response.text or "I'm having trouble with that."
        self.conversation.current.add_assistant(final_reply)
        await self.conversation.save()
        _log.info("conversation.reply", text=final_reply)
        await self.bus.publish(BotReply(text=final_reply, user_text=user_text))
        await self._speak_reply(final_reply)

    # ------------------------------------------------------------------ streaming LLM
    async def _handle_streaming(
        self,
        user_text: str,
        memory_context: str,
        tool_schemas: list[dict[str, Any]] | None,
    ) -> None:
        """Handle a conversation turn using a streaming LLM call."""
        self.conversation.current.add_user(user_text)
        messages = self.conversation.messages_for(memory_context)

        for round_num in range(_MAX_TOOL_ROUNDS):
            reply_parts: list[str] = []
            accumulated_tool_calls: list[ToolCall] = []

            async for chunk in self.llm.stream_complete(messages, tools=tool_schemas):  # type: ignore[attr-defined]
                if chunk.token:
                    reply_parts.append(chunk.token)
                    await self.bus.publish(LLMTokenReceived(token=chunk.token, done=False))
                if chunk.tool_calls:
                    accumulated_tool_calls.extend(chunk.tool_calls)
                if chunk.done:
                    break

            if accumulated_tool_calls:
                # Dispatch tool calls and re-call the LLM with results.
                tool_results = await self._execute_tool_calls(accumulated_tool_calls)
                text_so_far = "".join(reply_parts)
                messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        content=text_so_far,
                        tool_calls=tuple(accumulated_tool_calls),
                    )
                )
                for tc, result in zip(accumulated_tool_calls, tool_results, strict=False):
                    messages.append(Message(role=Role.TOOL, content=result, tool_call_id=tc.id))
                _log.info(
                    "conversation.streaming_tool_round",
                    calls=len(accumulated_tool_calls),
                    round=round_num,
                )
                continue

            # No tool calls - final response.
            reply = "".join(reply_parts)
            self.conversation.current.add_assistant(reply)
            await self.conversation.save()
            self._remember_exchange(user_text, reply)

            _log.info("conversation.reply", text=reply)
            await self.bus.publish(BotReply(text=reply, user_text=user_text))
            await self._speak_reply(reply)
            return

        # Exhausted tool rounds.
        final_reply = "".join(reply_parts) or "I'm having trouble with that."
        self.conversation.current.add_assistant(final_reply)
        await self.conversation.save()
        _log.info("conversation.reply", text=final_reply)
        await self.bus.publish(BotReply(text=final_reply, user_text=user_text))
        await self._speak_reply(final_reply)

    # ------------------------------------------------------------------ tool execution
    async def _execute_tool_calls(
        self, tool_calls: list[ToolCall] | tuple[ToolCall, ...]
    ) -> list[str]:
        """Execute a list of tool calls and return their JSON results."""
        if self.tool_executor is None:
            _log.warning("conversation.no_tool_executor")
            return ['{"error": "no tool executor available"}'] * len(tool_calls)

        results: list[str] = []
        for tc in tool_calls:
            try:
                result = await self.tool_executor.execute_tool_call(tc.name, tc.arguments)
                results.append(json.dumps(result))
            except Exception as exc:
                _log.exception("conversation.tool_call_failed", tool=tc.name, id=tc.id)
                results.append(json.dumps({"error": str(exc), "tool": tc.name}))
        return results

    # ------------------------------------------------------------------ helpers
    async def _on_state_change(self, event: StateChanged) -> None:
        _log.debug(
            "conversation.state_change",
            previous=event.previous.value,
            current=event.current.value,
        )

    def _memory_context(self, user_text: str) -> str:
        """Format relevant prior memories and preferences for safe system-prompt injection.

        When ``memory`` is a :class:`VectorMemory`, uses semantic search
        (:meth:`VectorMemory.search_similar`) for better recall. Falls
        back to keyword search (:meth:`Memory.search`) for the base
        :class:`Memory` class.

        When ``preference_tracker`` is set, appends the learned
        preferences to the context string.
        """
        parts: list[str] = []

        if self.memory is not None:
            memory_text = ""
            try:
                if isinstance(self.memory, VectorMemory):
                    results = self.memory.search_similar(
                        user_text, limit=self.memory_recall_limit, min_similarity=0.0
                    )
                    memory_text = "\n".join(f"- {entry.content}" for entry, _sim in results)
                else:
                    matches = self.memory.search(user_text)
                    entries = matches[-self.memory_recall_limit :] or self.memory.recall(
                        self.memory_recall_limit
                    )
                    memory_text = "\n".join(f"- {entry.content}" for entry in entries)
            except Exception:
                _log.warning("conversation.memory_context_failed", text=user_text[:80])
            if memory_text:
                parts.append(memory_text)

        if self.preference_tracker is not None:
            pref_text = self.preference_tracker.format_for_prompt()
            if pref_text:
                parts.append(pref_text)

        return "\n\n".join(parts)

    def _remember_exchange(self, user_text: str, reply: str) -> None:
        """Store compact facts from the completed exchange for later recall.

        Memory failures (e.g. embedding model errors) must not crash
        the conversation.  Errors are logged and swallowed so the
        BotReply / TTS path continues uninterrupted.
        """
        if self.memory is None:
            return
        import contextlib

        with contextlib.suppress(Exception):
            self.memory.add(f"User said: {user_text}", importance=0.7, tags=("user",))
        with contextlib.suppress(Exception):
            self.memory.add(f"DeskBot replied: {reply}", importance=0.3, tags=("assistant",))

    def _prepare_chunk_for_wake(self, chunk: AudioChunk) -> AudioChunk:
        """Resample and down-mix audio explicitly when the wake checker requires it."""
        target_rate = int(getattr(self.wake_checker, "sample_rate", chunk.sample_rate))
        target_channels = 1
        if chunk.sample_rate == target_rate and chunk.channels == target_channels:
            return chunk
        converted = convert_audio(
            AudioBuffer(
                pcm=chunk.pcm,
                sample_rate=chunk.sample_rate,
                channels=chunk.channels,
            ),
            target_sample_rate=target_rate,
            target_channels=target_channels,
        )
        # _log.info(
        #         #     "conversation.audio_chunk.normalized",
        return AudioChunk(
            pcm=converted.pcm,
            sample_rate=converted.sample_rate,
            channels=converted.channels,
            timestamp=chunk.timestamp,
        )

    async def _speak_reply(self, text: str) -> None:
        """Run TTS synthesis and audio playback, keeping SPEAKING active until done.

        The flow is:

        1. Transition to SPEAKING.
        2. Synthesise audio via ``tts.speak()`` -> ``AudioBuffer``.
        3. Play the buffer through ``self.audio`` (the configured
           :class:`AudioOutput`).
        4. Transition back to IDLE only after playback completes (or
           fails).

        If the TTS backend is :class:`MockTTS` or the audio output is
        :class:`MockAudioOutput`, degradation is logged clearly so a
        production run can never mistake "speech was generated" for
        "speech was physically played."
        """
        await self.state_machine.transition(RobotState.SPEAKING)
        mock_tts = isinstance(self.tts, MockTTS)
        mock_audio = self._is_mock_audio()
        try:
            _log.info(
                "tts.synthesis.started",
                backend=type(self.tts).__name__,
                text_length=len(text),
                mock_tts=mock_tts,
            )
            buffer = await self.tts.speak(text)
            _log.info(
                "tts.synthesis.completed",
                backend=type(self.tts).__name__,
                bytes=len(buffer.pcm),
                sample_rate=buffer.sample_rate,
                channels=buffer.channels,
                mock_tts=mock_tts,
            )
            if mock_tts:
                _log.warning(
                    "audio.playback.mock_tts",
                    tts_backend=type(self.tts).__name__,
                    message="text was accepted by MockTTS; no physical speech was produced",
                )
                return
            if buffer.is_empty:
                _log.warning(
                    "tts.synthesis.empty_buffer",
                    backend=type(self.tts).__name__,
                    message="TTS returned an empty AudioBuffer; no speech to play",
                )
                return
            if self.audio is None:
                _log.warning(
                    "audio.playback.no_output",
                    tts_backend=type(self.tts).__name__,
                    message="TTS produced audio but no AudioOutput is configured; "
                    "no physical speech will be heard",
                )
                return
            if mock_audio:
                _log.warning(
                    "audio.playback.mock_audio",
                    audio_backend=type(self.audio).__name__,
                    message="audio fell back to MockAudioOutput; no physical speech will be heard",
                )
                # Still call play() so tests can verify the buffer reaches
                # the output, but it will produce no physical sound.
            _log.info(
                "audio.playback.started",
                audio_backend=type(self.audio).__name__,
                tts_backend=type(self.tts).__name__,
                bytes=len(buffer.pcm),
                sample_rate=buffer.sample_rate,
                channels=buffer.channels,
                mock_audio=mock_audio,
            )
            await self.audio.play(buffer)
            _log.info(
                "audio.playback.completed",
                audio_backend=type(self.audio).__name__,
                tts_backend=type(self.tts).__name__,
                bytes=len(buffer.pcm),
                sample_rate=buffer.sample_rate,
                channels=buffer.channels,
                duration_s=round(buffer.duration_s, 3),
            )
        except Exception:
            _log.exception("conversation.speech_output_failed")
        finally:
            await self.state_machine.transition(RobotState.IDLE)

    def _is_mock_audio(self) -> bool:
        """Return True when the audio output is a mock that produces no sound."""
        if self.audio is None:
            return False
        return type(self.audio).__name__ == "MockAudioOutput"


def build_default_conversation(
    bus: InMemoryEventBus, state_machine: StateMachine
) -> ConversationService:
    """Helper that builds a fully-wired :class:`ConversationService`."""
    stt = MockSTT()
    tts = MockTTS()
    llm = MockLLM()
    llm.register("hello", "Hi there!")
    llm.register("how are you", "Feeling very electric today.")
    conversation = ConversationManager(llm=llm, system_prompt=system_prompt())
    return ConversationService(
        bus=bus,
        state_machine=state_machine,
        stt=stt,
        tts=tts,
        llm=llm,
        conversation=conversation,
    )


__all__ = ["ConversationService", "build_default_conversation"]
