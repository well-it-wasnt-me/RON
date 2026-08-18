"""Transition lifecycle: open/close real state-action-outcome transitions.

A transition is not a single synchronous call — it is a lifecycle:

::

    OBSERVE state_t
        |
        v
    transition_store.begin(state=state_t, action_index=Y)
        |
        v
    EXECUTE action Y  (robot moves / acts)
        |
        v
    OBSERVE state_t+1
        |
        v
    transition.complete(next_state=state_t+1, reward=R, done=...)
        |
        v
    STORE completed transition  →  Experience

Until ``complete()`` is called, no experience is persisted.  This
prevents the class of bugs where the recorder encodes two consecutive
states without an intervening action (fake transitions).

Every transition carries a real action identity taken from the
configured :class:`ActionSpace`.  Observation events (``FaceDetected``,
``SpeechRecognized``, …) are **not** actions and must never appear as
the action field of a transition.

Execution metadata (execution id, action id, start/completion
timestamps, success/failure, latency, policy version) is recorded
so a stored transition can be fully reconstructed:

::

    At T0:  observation = X
    At T0:  policy selected action = Y
           Robot executed Y
    At T1:  observation = Z
           Reward = R

Malformed transitions (missing action, invalid action, NaN/inf,
impossible timestamps, missing next state) are rejected.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from robot.learning.action_learning import ActionSpace
from robot.learning.experience import Experience
from robot.learning.observation import Observation
from robot.logging import get_logger

_log = get_logger("learning.transition")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TransitionError(ValueError):
    """Raised when a transition is malformed or violates lifecycle rules."""


# ---------------------------------------------------------------------------
# Completed transition
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Transition:
    """A completed state → action → next_state → reward transition.

    Attributes
    ----------
    transition_id:
        Unique identifier for this transition.
    execution_id:
        Identifier of the hardware execution that produced the
        next state (may be the same as transition_id when not
        provided).
    state:
        Observation vector *before* the action was executed.
    action_index:
        Index of the selected action in the :class:`ActionSpace`.
    action_name:
        Human-readable name of the selected action.
    action_vector:
        One-hot encoding of the action (size = ``action_space.size``).
    reward:
        Scalar reward computed *after* the outcome was observed.
    next_state:
        Observation vector *after* the action was executed.
    done:
        Whether the episode/task terminated after this transition.
    start_timestamp_ns:
        Monotonic nanosecond timestamp when the transition was opened
        (``begin()`` was called).
    completion_timestamp_ns:
        Monotonic nanosecond timestamp when the transition was
        completed (``complete()`` was called).
    execution_success:
        ``True`` if the action executed successfully on hardware.
    execution_failure_reason:
        Empty string on success; a human-readable reason on failure.
    latency_ms:
        Wall-clock latency between begin and complete, in milliseconds.
    policy_version:
        Version string of the policy that selected the action
        (``"deterministic"`` for the built-in controller).
    metadata:
        Additional key-value pairs forwarded from the caller.
    """

    transition_id: str
    execution_id: str
    state: list[float]
    action_index: int
    action_name: str
    action_vector: list[float]
    reward: float
    next_state: list[float]
    done: bool
    start_timestamp_ns: int
    completion_timestamp_ns: int
    execution_success: bool
    execution_failure_reason: str
    latency_ms: float
    policy_version: str
    observation: Observation | None = None
    next_observation: Observation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def to_experience(self) -> Experience:
        """Convert to an :class:`Experience` for storage in memory layers.

        The action vector from the ActionSpace is used as the action
        field so downstream models consume a fixed-size vector.  All
        transition-specific metadata is preserved in ``metadata``.
        """
        merged: dict[str, Any] = {
            "transition_id": self.transition_id,
            "execution_id": self.execution_id,
            "action_index": self.action_index,
            "action_name": self.action_name,
            "start_timestamp_ns": self.start_timestamp_ns,
            "completion_timestamp_ns": self.completion_timestamp_ns,
            "execution_success": self.execution_success,
            "execution_failure_reason": self.execution_failure_reason,
            "latency_ms": self.latency_ms,
            "policy_version": self.policy_version,
        }
        merged.update(self.metadata)
        return Experience(
            timestamp=datetime.now(tz=UTC),
            state=list(self.state),
            action=list(self.action_vector),
            reward=self.reward,
            next_state=list(self.next_state),
            metadata=merged,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "transition_id": self.transition_id,
            "execution_id": self.execution_id,
            "state": list(self.state),
            "action_index": self.action_index,
            "action_name": self.action_name,
            "action_vector": list(self.action_vector),
            "reward": self.reward,
            "next_state": list(self.next_state),
            "done": self.done,
            "start_timestamp_ns": self.start_timestamp_ns,
            "completion_timestamp_ns": self.completion_timestamp_ns,
            "execution_success": self.execution_success,
            "execution_failure_reason": self.execution_failure_reason,
            "latency_ms": self.latency_ms,
            "policy_version": self.policy_version,
            "observation": self.observation.to_dict() if self.observation else None,
            "next_observation": self.next_observation.to_dict() if self.next_observation else None,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Pending (open) transition
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PendingTransition:
    """An open transition that has not yet been completed.

    Created by :meth:`TransitionStore.begin`.  Call :meth:`complete` to
    close it after the action has been executed and the outcome
    observed.
    """

    transition_id: str
    execution_id: str
    state: list[float]
    action_index: int
    action_name: str
    action_vector: list[float]
    start_timestamp_ns: int
    policy_version: str
    observation: Observation | None = None
    _store: TransitionStore | None = field(default=None, repr=False)
    _completed: bool = field(default=False, init=False, repr=False)

    @property
    def is_completed(self) -> bool:
        return self._completed

    def complete(
        self,
        next_state: list[float],
        reward: float,
        done: bool = False,
        execution_success: bool = True,
        execution_failure_reason: str = "",
        metadata: dict[str, Any] | None = None,
        next_observation: Observation | None = None,
    ) -> Transition:
        """Close this transition and store the completed experience.

        Parameters
        ----------
        next_state:
            Observation vector *after* the action was executed.
        reward:
            Scalar reward computed after observing the outcome.
        done:
            Whether the episode terminated.
        execution_success:
            Whether the hardware action executed successfully.
        execution_failure_reason:
            Human-readable reason when ``execution_success`` is False.
        metadata:
            Additional metadata forwarded into the stored transition.

        Returns
        -------
        Transition
            The completed, immutable transition.

        Raises
        ------
        TransitionError
            If the transition is already completed or the payload is
            malformed.
        """
        assert self._store is not None
        return self._store._complete(
            self,
            next_state=next_state,
            reward=reward,
            done=done,
            execution_success=execution_success,
            execution_failure_reason=execution_failure_reason,
            metadata=metadata,
            next_observation=next_observation,
        )


# ---------------------------------------------------------------------------
# Transition store
# ---------------------------------------------------------------------------


def _validate_vector(vec: list[float], name: str) -> None:
    """Reject NaN, inf, or empty vectors."""
    if not vec:
        raise TransitionError(f"{name} must not be empty")
    for i, v in enumerate(vec):
        f = float(v)
        if math.isnan(f):
            raise TransitionError(f"{name}[{i}] is NaN")
        if math.isinf(f):
            raise TransitionError(f"{name}[{i}] is inf")


@dataclass(slots=True)
class TransitionStore:
    """Manages the open→closed transition lifecycle.

    Parameters
    ----------
    action_space:
        The action space actions are selected from.  Every transition
        must reference a valid action index from this space.
    on_transition_completed:
        Optional callback invoked after each transition is completed
        and stored.  Receives the :class:`Transition` as its sole
        argument.  Typically this stores the resulting
        :class:`Experience` in working/replay/episodic memory.
    """

    action_space: ActionSpace
    on_transition_completed: Callable[[Transition], None] | None = None
    _pending: dict[str, PendingTransition] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------ begin
    def begin(
        self,
        state: list[float],
        action_index: int,
        execution_id: str | None = None,
        policy_version: str = "deterministic",
    ) -> PendingTransition:
        """Open a new transition.

        Validates that the action index belongs to the configured
        action space and that the state vector is well-formed.  No
        experience is stored yet.

        Parameters
        ----------
        state:
            Observation vector *before* the action is executed.
        action_index:
            Index of the selected action in :attr:`action_space`.
        execution_id:
            Optional identifier for the hardware execution.  A UUID
            is generated when not provided.
        policy_version:
            Version string of the policy that selected the action.

        Returns
        -------
        PendingTransition
            The open transition; call ``.complete()`` after execution.

        Raises
        ------
        TransitionError
            If the action index is invalid or the state is malformed.
        """
        _validate_vector(state, "state")

        if not (0 <= action_index < self.action_space.size):
            raise TransitionError(
                f"action_index {action_index} is out of range [0, {self.action_space.size})"
            )

        action = self.action_space.get(action_index)
        action_vector = self.action_space.action_vector(action_index).tolist()

        transition_id = str(uuid.uuid4())
        exec_id = execution_id or transition_id

        pending = PendingTransition(
            transition_id=transition_id,
            execution_id=exec_id,
            state=list(state),
            action_index=action_index,
            action_name=action.name,
            action_vector=action_vector,
            start_timestamp_ns=time.monotonic_ns(),
            policy_version=policy_version,
            _store=self,
        )
        self._pending[transition_id] = pending
        return pending

    def begin_observation(
        self,
        observation: Observation,
        action_index: int,
        execution_id: str | None = None,
        policy_version: str = "deterministic",
    ) -> PendingTransition:
        """Open a transition from a typed :class:`Observation`.

        Like :meth:`begin` but takes an :class:`Observation` instead
        of a raw float vector.  The observation is encoded to a vector
        for the ``state`` field and also stored directly for later
        inspection.

        Parameters
        ----------
        observation:
            The observation before the action is executed.
        action_index:
            Index of the selected action in :attr:`action_space`.
        execution_id:
            Optional identifier for the hardware execution.
        policy_version:
            Version string of the policy that selected the action.

        Returns
        -------
        PendingTransition
            The open transition with ``observation`` populated.
        """
        state = observation.to_vector()
        pending = self.begin(
            state=state,
            action_index=action_index,
            execution_id=execution_id,
            policy_version=policy_version,
        )
        # Attach the typed observation
        object.__setattr__(pending, "observation", observation)
        return pending

    # ------------------------------------------------------------------ complete
    def _complete(
        self,
        pending: PendingTransition,
        next_state: list[float],
        reward: float,
        done: bool,
        execution_success: bool,
        execution_failure_reason: str,
        metadata: dict[str, Any] | None,
        next_observation: Observation | None = None,
    ) -> Transition:
        """Internal: close a pending transition and store it."""
        if pending._completed:
            raise TransitionError(f"transition {pending.transition_id} is already completed")
        if pending.transition_id not in self._pending:
            raise TransitionError(f"transition {pending.transition_id} not found in store")

        _validate_vector(next_state, "next_state")

        r = float(reward)
        if math.isnan(r) or math.isinf(r):
            raise TransitionError(f"reward is NaN or inf: {r}")

        completion_ns = time.monotonic_ns()
        if completion_ns < pending.start_timestamp_ns:
            raise TransitionError("completion timestamp is before start timestamp (non-monotonic)")

        latency_ms = (completion_ns - pending.start_timestamp_ns) / 1e6

        transition = Transition(
            transition_id=pending.transition_id,
            execution_id=pending.execution_id,
            state=list(pending.state),
            action_index=pending.action_index,
            action_name=pending.action_name,
            action_vector=list(pending.action_vector),
            reward=r,
            next_state=list(next_state),
            done=bool(done),
            start_timestamp_ns=pending.start_timestamp_ns,
            completion_timestamp_ns=completion_ns,
            execution_success=bool(execution_success),
            execution_failure_reason=execution_failure_reason or "",
            latency_ms=latency_ms,
            policy_version=pending.policy_version,
            observation=pending.observation,
            next_observation=next_observation,
            metadata=dict(metadata) if metadata else {},
        )

        pending._completed = True
        del self._pending[pending.transition_id]

        if self.on_transition_completed is not None:
            self.on_transition_completed(transition)

        _log.debug(
            "transition.completed",
            transition_id=transition.transition_id,
            action=transition.action_name,
            reward=round(transition.reward, 4),
            latency_ms=round(transition.latency_ms, 2),
            success=transition.execution_success,
        )

        return transition

    # ------------------------------------------------------------------ helpers
    @property
    def pending_count(self) -> int:
        """Number of open (not yet completed) transitions."""
        return len(self._pending)

    def record(
        self,
        state: list[float],
        action_index: int,
        next_state: list[float],
        reward: float,
        done: bool = False,
        execution_success: bool = True,
        execution_failure_reason: str = "",
        execution_id: str | None = None,
        policy_version: str = "deterministic",
        metadata: dict[str, Any] | None = None,
        observation: Observation | None = None,
        next_observation: Observation | None = None,
    ) -> Transition:
        """Begin and immediately complete a transition in one call.

        Convenience for cases where the caller already has the full
        transition tuple (e.g. loading from persistent storage or
        simulation).  Still validates everything.
        """
        pending = self.begin(
            state=state,
            action_index=action_index,
            execution_id=execution_id,
            policy_version=policy_version,
        )
        if observation is not None:
            object.__setattr__(pending, "observation", observation)
        return pending.complete(
            next_state=next_state,
            reward=reward,
            done=done,
            execution_success=execution_success,
            execution_failure_reason=execution_failure_reason,
            metadata=metadata,
            next_observation=next_observation,
        )


__all__ = [
    "PendingTransition",
    "Transition",
    "TransitionError",
    "TransitionStore",
]
