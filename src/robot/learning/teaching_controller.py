"""Teaching controller: drives the demonstrate / practice loop from gestures.

The controller ties together the pieces built in Phases 3-7 into the human
teaching loop the project targets:

1. The human says ``"RON, when I wave, wave back"`` — the constrained
   :func:`~robot.learning.teaching_parser.parse_teaching_instruction` turns that
   into a :class:`DemonstrationSpec` and :meth:`start_session` arms the
   controller (no LLM in the loop).
2. Each time a :class:`~robot.events.events.GestureDetected` event fires whose
   gesture matches the spec's trigger, :meth:`on_gesture_detected` opens an
   interaction and either **demonstrates** the desired action (executing it
   through the canonical :class:`ActionExecutor`, so it is recorded as a real
   transition) or, in **practice** mode, asks the policy
   (:class:`ActionLearner`) to propose an action and executes *that* — gated by
   the :class:`SafetyGate`.

Safety invariants
-----------------
* Every executed action flows through the same :class:`ActionExecutor` (the
  single learning recording point) — the controller never writes to hardware
  or the replay buffer directly.
* Practice proposals pass the non-mutating :class:`SafetyGateValidator` during
  selection and are re-validated with the full, mutating
  :meth:`SafetyGate.validate` before execution. An out-of-range or
  rate-limited proposal is rejected and turned into a no-op (fallback) — the
  controller never raises a ``ServoError``.
* The controller never invents a reward, mints a transition id, or bypasses
  the recorder. It only opens/closes the interaction context that tags the
  transitions the executor already records.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from robot.learning.action_mapping import action_index_to_behavior_action
from robot.learning.safety_gate import SafetyGateValidator
from robot.learning.teaching_parser import DemonstrationSpec, parse_teaching_instruction
from robot.logging import get_logger

if TYPE_CHECKING:
    from robot.learning.action_learning import ActionLearner, ActionSpace
    from robot.learning.interaction_context import InteractionContext
    from robot.learning.safety_gate import SafetyGate
    from robot.services.executor import ActionExecutor

_log = get_logger("learning.teaching_controller")


class TeachingController:
    """Coordinates a single active teaching session.

    Parameters
    ----------
    action_learner:
        The trained policy. ``None`` disables practice mode (the controller
        falls back to demonstration / no-op).
    safety_gate:
        The safety gate that gates every practice proposal.
    action_space:
        The action space used to resolve indices to behaviour actions.
    interaction_context:
        The shared interaction context that tags recorded transitions with a
        teaching session / interaction id.
    executor:
        The canonical action executor — the single point that records real
        transitions and drives hardware.
    min_experiences_for_practice:
        Minimum total experiences before practice mode is allowed to propose
        (the policy needs something to have learned from). Below this,
        practice falls back to demonstration.
    """

    def __init__(
        self,
        *,
        action_learner: ActionLearner | None,
        safety_gate: SafetyGate,
        action_space: ActionSpace,
        interaction_context: InteractionContext,
        executor: ActionExecutor,
        min_experiences_for_practice: int = 64,
    ) -> None:
        self._action_learner = action_learner
        self._safety_gate = safety_gate
        self._action_space = action_space
        self._interaction_context = interaction_context
        self._executor = executor
        self._min_experiences_for_practice = int(min_experiences_for_practice)

        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._current: DemonstrationSpec | None = None
        self._mode: str = "demonstrate"

    # ------------------------------------------------------------------ state
    @property
    def in_teaching_mode(self) -> bool:
        """Whether a teaching session is currently armed."""
        with self._lock:
            return self._current is not None

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    @property
    def current(self) -> DemonstrationSpec | None:
        with self._lock:
            return self._current

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    # ------------------------------------------------------------------ session
    def arm_from_instruction(self, text: str, mode: str = "demonstrate") -> str | None:
        """Parse a spoken teaching instruction and arm a session if it matches.

        Thin wrapper around
        :func:`~robot.learning.teaching_parser.parse_teaching_instruction` using
        this controller's action space. Returns the new session id, or ``None``
        when ``text`` is not a constrained teaching instruction (so the caller
        lets the utterance fall through to the normal conversation turn).
        """
        spec = parse_teaching_instruction(text, self._action_space)
        if spec is None:
            return None
        return self.start_session(spec, mode=mode)

    def start_session(self, spec: DemonstrationSpec, mode: str = "demonstrate") -> str:
        """Arm a teaching session for ``spec`` in ``mode``.

        Mints a teaching-session id on the shared interaction context so every
        transition the executor records during this session is tagged with it.
        Returns the session id.
        """
        if mode not in {"demonstrate", "practice"}:
            raise ValueError(f"unknown teaching mode: {mode!r}")
        with self._lock:
            self._current = spec
            self._mode = mode
        session_id = self._interaction_context.begin_teaching_session()
        with self._lock:
            self._session_id = session_id
        _log.info(
            "teaching.session_started",
            session_id=session_id,
            mode=mode,
            trigger=spec.trigger_gesture,
            action=spec.desired_action,
        )
        return session_id

    def end_session(self) -> None:
        """Disarm the teaching session and clear the interaction context."""
        with self._lock:
            self._current = None
            self._session_id = None
            self._mode = "demonstrate"
        self._interaction_context.end_teaching_session()
        _log.info("teaching.session_ended")

    # ------------------------------------------------------------------ gesture
    async def on_gesture_detected(
        self,
        gesture: str,
        state: Sequence[float] | np.ndarray,
    ) -> int | None:
        """Handle a ``GestureDetected`` event.

        If a session is armed and ``gesture`` matches the spec's trigger,
        open an interaction and execute one action (demonstrate or practice),
        then close the interaction. Returns the executed action index, or
        ``None`` if the gesture did not trigger, the proposal was rejected by
        the safety gate, or the resolved action was invalid.
        """
        with self._lock:
            spec = self._current
            mode = self._mode
        if spec is None or gesture != spec.trigger_gesture:
            return None

        # Normalise to an ndarray for the policy / safety gate.
        state_arr = np.asarray(state, dtype=float)
        self._interaction_context.begin_interaction()
        try:
            if mode == "practice":
                return await self._practice(state_arr)
            return await self._demonstrate(spec)
        finally:
            self._interaction_context.end_interaction()

    # ------------------------------------------------------------------ modes
    async def _demonstrate(self, spec: DemonstrationSpec) -> int | None:
        """Execute the human-specified desired action through the executor."""
        action = action_index_to_behavior_action(spec.desired_action_index, self._action_space)
        if action is None:
            _log.warning(
                "teaching.demonstrate_unresolvable",
                action_index=spec.desired_action_index,
            )
            return None
        await self._executor.execute_one(action)
        return spec.desired_action_index

    async def _practice(self, state: np.ndarray) -> int | None:
        """Let the policy propose an action, gated by the safety gate.

        Falls back to demonstration when there is no learner or too few
        experiences to trust a proposal. The proposal is re-validated with the
        full (mutating) safety gate before execution; a rejected proposal
        becomes a no-op rather than an unsafe execution.
        """
        if self._action_learner is None:
            _log.debug("teaching.practice_no_learner")
            return None

        # Below the experience floor, the policy has not learned enough —
        # demonstrate the desired action instead of trusting a near-random
        # proposal. (Practice requires action_learner is not None.)
        spec = self.current
        if spec is not None and self._total_experiences() < self._min_experiences_for_practice:
            return await self._demonstrate(spec)

        validator = SafetyGateValidator(self._safety_gate)
        proposed = self._action_learner.select_action(state, validator=validator)

        # Defense-in-depth: re-validate the single chosen action with the full,
        # mutating gate (cooldown / rate / static range) before execution.
        params = None
        if 0 <= proposed < self._action_space.size:
            params = self._action_space.get(proposed).params
        gate_result = self._safety_gate.validate(proposed, state=list(state), params=params)
        if not gate_result.allowed:
            _log.info(
                "teaching.practice_rejected",
                proposed=proposed,
                reason=gate_result.reason,
            )
            return None

        action = action_index_to_behavior_action(proposed, self._action_space)
        if action is None:
            _log.warning("teaching.practice_unresolvable", action_index=proposed)
            return None
        await self._executor.execute_one(action)
        return proposed

    def _total_experiences(self) -> int:
        """Total experiences the policy has trained on (0 if unavailable)."""
        learner = self._action_learner
        # The ActionLearner exposes step_count; the LearningService exposes
        # status.total_experiences. We probe the latter via the executor's
        # recorder when wired, else fall back to the learner's step count.
        recorder = getattr(self._executor, "experience_recorder", None)
        working = getattr(recorder, "working_memory", None) if recorder is not None else None
        if working is not None:
            try:
                return len(working.recent(10_000))
            except Exception:
                pass
        step = getattr(learner, "step_count", None)
        if isinstance(step, int):
            return step
        return 0


__all__ = ["TeachingController"]
