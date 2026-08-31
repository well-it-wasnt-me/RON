"""Experience recorder that bridges DeskBot events to learning memory.

The recorder subscribes to the event bus and keeps the :class:`StateEncoder`
in sync with the robot's current observation.  Crucially, **observation
events** (``FaceDetected``, ``SpeechRecognized``, ``EmotionChanged``,
``IdleTimeout``, …) update the encoder state but do **not** create
transitions.  Only a real action selected from the :class:`ActionSpace`
and executed on the robot produces a transition via the
:class:`TransitionStore` lifecycle:

::

    OBSERVE state_t
        |
    transition_store.begin(state=state_t, action_index=Y)
        |
    EXECUTE action Y
        |
    OBSERVE state_t+1
        |
    pending.complete(next_state=state_t+1, reward=R, done=…)
        |
    STORE completed transition -> Experience

This prevents the class of fake transitions where two consecutive
states are encoded without an intervening action.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from robot.events.bus import InMemoryEventBus
from robot.events.events import (
    EmotionChanged,
    FaceDetected,
    GestureDetected,
    IdleTimeout,
    ServoMoved,
    SpeechRecognized,
    StateChanged,
)
from robot.learning.action_learning import ActionSpace, deskbot_action_space
from robot.learning.experience import EpisodicMemory, Experience, ReplayBuffer, WorkingMemory
from robot.learning.observation import Observation
from robot.learning.reward import RewardModel
from robot.learning.state_encoder import StateEncoder
from robot.learning.transition import PendingTransition, Transition, TransitionStore

if TYPE_CHECKING:
    from robot.learning.multimodal import MultimodalEncoder
from robot.logging import get_logger

_log = get_logger("learning.recorder")


@dataclass(slots=True)
class ExperienceRecorder:
    """Bridges DeskBot events to experience memory.

    Subscribes to the event bus and keeps the encoder in sync with the
    robot's current observation.  Observation events update the encoder
    state; only real actions produce transitions through the
    :class:`TransitionStore`.

    Parameters
    ----------
    bus:
        The event bus to subscribe to.
    action_space:
        The action space actions are selected from.  Defaults to
        :func:`deskbot_action_space`.
    encoder:
        The state encoder. If None, a default one is created.
    working_memory:
        Short-term ring buffer.
    replay_buffer:
        Training replay buffer.
    episodic_memory:
        Persistent episodic memory (may be None if persistence is
        disabled).
    default_reward:
        Default reward for transitions where no explicit reward is
        provided.
    on_experience_recorded:
        Optional callback invoked after each experience is stored.
        Receives the :class:`Experience` as its sole argument.
    """

    bus: InMemoryEventBus
    action_space: ActionSpace = field(default_factory=deskbot_action_space)
    encoder: StateEncoder = field(default_factory=StateEncoder)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    replay_buffer: ReplayBuffer = field(default_factory=ReplayBuffer)
    episodic_memory: EpisodicMemory | None = None
    default_reward: float = 0.0
    reward_model: RewardModel = field(default_factory=RewardModel)
    on_experience_recorded: Callable[[Experience], None] | None = field(default=None, repr=False)
    #: Optional multimodal encoder. When set, transitions use the multimodal
    #: encode() for state vectors while event handlers still update the inner
    #: StateEncoder (accessible via multimodal_encoder.state_encoder).
    multimodal_encoder: MultimodalEncoder | None = field(default=None, init=False, repr=False)

    # Transition store (created in __post_init__)
    transition_store: TransitionStore = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _subscribed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.transition_store = TransitionStore(
            action_space=self.action_space,
            on_transition_completed=self._on_transition_completed,
        )

    def _encode_state(self) -> list[float]:
        """Produce the state vector for transition recording.

        When a multimodal encoder is configured, uses its richer encode()
        (trainable vision/audio sub-encoders + temporal history). Otherwise
        falls back to the plain StateEncoder.
        """
        if self.multimodal_encoder is not None:
            return self.multimodal_encoder.encode()
        return self.encoder.encode()

    # ------------------------------------------------------------------ lifecycle
    def attach(self) -> None:
        """Subscribe to the event bus."""
        if self._subscribed:
            return
        self.bus.subscribe(StateChanged, self._on_state_changed)
        self.bus.subscribe(EmotionChanged, self._on_emotion_changed)
        self.bus.subscribe(ServoMoved, self._on_servo_moved)
        self.bus.subscribe(FaceDetected, self._on_face_detected)
        # GestureDetected is an observation (synthetic injection channel today,
        # no built-in CV detector). It updates the gesture one-hot / person
        # flag in the encoder but never creates a transition.
        self.bus.subscribe(GestureDetected, self._on_gesture_detected)
        # SpeechRecognized is an observation, not an action - not subscribed
        # to avoid paying bus dispatch cost for a no-op handler.
        self.bus.subscribe(IdleTimeout, self._on_idle_timeout)
        self._subscribed = True
        _log.info("experience_recorder.attached")

    def detach(self) -> None:
        """Unsubscribe from the event bus."""
        self.bus.unsubscribe(StateChanged, self._on_state_changed)
        self.bus.unsubscribe(EmotionChanged, self._on_emotion_changed)
        self.bus.unsubscribe(ServoMoved, self._on_servo_moved)
        self.bus.unsubscribe(FaceDetected, self._on_face_detected)
        self.bus.unsubscribe(GestureDetected, self._on_gesture_detected)
        self.bus.unsubscribe(IdleTimeout, self._on_idle_timeout)
        self._subscribed = False

    # ------------------------------------------------------------------ context
    def update_context(self, **kwargs: Any) -> None:
        """Manually update the encoder context.

        Recognised keys mirror :class:`StateEncoder` fields. Unknown keys
        are ignored. Dict fields (``emotions``/``servos``/``personality``)
        are *merged* into the existing mapping; ``vision``/``audio``/
        ``idle_seconds`` replace the field directly. Teaching/gesture keys
        are forwarded to the encoder's typed updaters.
        """
        # Dict fields are merged (in place) to match the legacy behaviour.
        merge_keys = ("emotions", "servos", "personality")
        # Scalar/object fields replace the encoder attribute directly.
        assign_keys = ("vision", "audio", "idle_seconds")
        # Typed-updater dispatch for fields with a dedicated method.
        updaters: dict[str, Callable[[Any], None]] = {
            "state": self.encoder.update_state,
            "teaching_context": self.encoder.update_teaching_context,
            "interaction_active": self.encoder.update_interaction_active,
            "person_present": self.encoder.update_person_present,
            "gesture": self.encoder.update_gesture,
            "conversation_turn": self.encoder.update_conversation_turn,
            "last_action_index": self.encoder.update_last_action,
        }
        for key, value in kwargs.items():
            if key in merge_keys:
                getattr(self.encoder, key).update(value)
            elif key in assign_keys:
                setattr(self.encoder, key, value)
            elif key in updaters:
                updaters[key](value)

    # ------------------------------------------------------------------ transition lifecycle
    def begin_transition(
        self,
        action_index: int,
        execution_id: str | None = None,
        policy_version: str = "deterministic",
    ) -> PendingTransition:
        """Open a transition by snapshotting the current state and action.

        The current encoder state is captured as ``state_t``.  After the
        action is executed on the robot and the outcome is observed, call
        :meth:`complete_transition` to close the transition.

        Parameters
        ----------
        action_index:
            Index of the selected action in the configured action space.
        execution_id:
            Optional identifier for the hardware execution.
        policy_version:
            Version string of the policy that selected the action.

        Returns
        -------
        PendingTransition
            The open transition.
        """
        state = self._encode_state()
        return self.transition_store.begin(
            state=state,
            action_index=action_index,
            execution_id=execution_id,
            policy_version=policy_version,
        )

    def begin_observation_transition(
        self,
        action_index: int,
        execution_id: str | None = None,
        policy_version: str = "deterministic",
    ) -> PendingTransition:
        """Open a transition from a typed :class:`Observation`.

        Captures the current encoder state as an :class:`Observation`
        snapshot, encodes it to a vector, and opens a transition.
        The typed observation is stored alongside the vector for later
        inspection.

        Parameters
        ----------
        action_index:
            Index of the selected action in the configured action space.
        execution_id:
            Optional identifier for the hardware execution.
        policy_version:
            Version string of the policy that selected the action.

        Returns
        -------
        PendingTransition
            The open transition with ``observation`` populated.
        """
        observation = Observation.from_encoder(self.encoder)
        return self.transition_store.begin_observation(
            observation=observation,
            action_index=action_index,
            execution_id=execution_id,
            policy_version=policy_version,
        )

    def complete_transition(
        self,
        pending: PendingTransition,
        reward: float | None = None,
        done: bool = False,
        execution_success: bool = True,
        execution_failure_reason: str = "",
        metadata: dict[str, Any] | None = None,
        use_reward_model: bool = False,
    ) -> Transition:
        """Close a pending transition after the action has been executed.

        Captures the current encoder state as ``state_t+1`` (the
        observation after the outcome).  The completed transition is
        stored as an :class:`Experience` in all configured memory layers
        and the ``on_experience_recorded`` callback is invoked.

        Parameters
        ----------
        pending:
            The open transition returned by :meth:`begin_transition`
            or :meth:`begin_observation_transition`.
        reward:
            Scalar reward.  When ``None`` and ``use_reward_model`` is
            False, :attr:`default_reward` is used.  When ``None`` and
            ``use_reward_model`` is True, the :class:`RewardModel`
            computes the reward from the observation/action/outcome.
        done:
            Whether the episode terminated.
        execution_success:
            Whether the action executed successfully on hardware.
        execution_failure_reason:
            Human-readable reason on failure.
        metadata:
            Additional metadata forwarded into the stored transition.
        use_reward_model:
            When True and ``reward`` is None, compute the reward using
            the :class:`RewardModel` instead of the default.
        """
        next_state = self._encode_state()

        # Capture typed observation if the transition has one
        next_observation = None
        if pending.observation is not None:
            next_observation = Observation.from_encoder(self.encoder)

        # Compute reward
        if reward is not None:
            r = reward
        elif use_reward_model and pending.observation is not None and next_observation is not None:
            r = self.reward_model.compute_for_action_index(
                observation=pending.observation,
                action_index=pending.action_index,
                next_observation=next_observation,
                action_space=self.action_space,
            )
        else:
            r = self.default_reward

        return pending.complete(
            next_state=next_state,
            reward=r,
            done=done,
            execution_success=execution_success,
            execution_failure_reason=execution_failure_reason,
            metadata=metadata,
            next_observation=next_observation,
        )

    # ------------------------------------------------------------------ manual record (legacy / validated)
    def record(
        self,
        state: list[float],
        action: list[float],
        reward: float,
        next_state: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> Experience:
        """Manually record an experience tuple.

        This bypasses the transition lifecycle and stores an experience
        directly.  It is retained for backward compatibility and for
        loading pre-collected data.  The action vector is stored as-is.

        .. deprecated::
            Prefer the transition lifecycle (:meth:`begin_transition` /
            :meth:`complete_transition`) which validates action
            identity.
        """
        exp = Experience(
            timestamp=datetime.now(tz=UTC),
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            metadata=metadata or {},
        )
        self._store(exp)
        return exp

    # ------------------------------------------------------------------ storage
    def _on_transition_completed(self, transition: Transition) -> None:
        """Store a completed transition as an Experience in all memory layers."""
        exp = transition.to_experience()
        self._store(exp)

    def _store(self, experience: Experience) -> None:
        """Store an experience in all memory layers and notify the callback."""
        self.working_memory.add(experience)
        self.replay_buffer.add(experience)
        if self.episodic_memory is not None:
            self.episodic_memory.add(experience)
        if self.on_experience_recorded is not None:
            self.on_experience_recorded(experience)

    # ------------------------------------------------------------------ handlers (observations only)
    async def _on_state_changed(self, event: StateChanged) -> None:
        """Observe robot state transitions - updates encoder only."""
        self.encoder.update_state(event.current)

    async def _on_emotion_changed(self, event: EmotionChanged) -> None:
        """Observe emotion changes - updates encoder only."""
        self.encoder.update_emotion(event.current, event.intensity)

    async def _on_servo_moved(self, event: ServoMoved) -> None:
        """Observe servo movements - updates encoder only."""
        self.encoder.update_servo(event.name, event.angle)

    async def _on_face_detected(self, event: FaceDetected) -> None:
        """Observe face detection - updates encoder only, no transition."""
        self.encoder.update_vision(
            face_detected=True,
            face_x=event.x,
            face_y=event.y,
            face_confidence=event.confidence,
            face_count=1,
        )
        # A detected face means a person is present (teaching context flag).
        self.encoder.update_person_present(True)

    async def _on_gesture_detected(self, event: GestureDetected) -> None:
        """Observe a gesture - updates encoder only, no transition.

        The gesture one-hot and person-present flag are updated. This is
        an observation, never an action; no transition is created here.
        """
        self.encoder.update_gesture(event.gesture)
        self.encoder.update_person_present(True)

    async def _on_speech_recognized(self, event: SpeechRecognized) -> None:
        """Observe speech recognition - updates encoder only, no transition."""
        # Speech recognition is an observation, not an action.
        # The encoder does not have a direct speech field, but we can
        # mark interaction state.  No transition is created.
        pass

    async def _on_idle_timeout(self, event: IdleTimeout) -> None:
        """Observe idle timeout - updates encoder only, no transition."""
        self.encoder.update_idle(event.seconds_idle)


__all__ = ["ExperienceRecorder"]
