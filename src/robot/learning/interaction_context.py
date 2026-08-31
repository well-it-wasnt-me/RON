"""Mutable context that tags learning transitions with a teaching interaction.

A teaching interaction is a bounded span of human-guided activity:

* ``interaction_id`` - one human <-> robot interaction (e.g. a teaching session).
* ``teaching_session_id`` - an explicit teaching session started by the
  developer (``"RON, when I wave, wave back"``).
* ``episode_id`` - a single demonstration/practice episode within a session
  (human waves -> robot waves -> human says "Good").

This module only **holds** the active identifiers. It never mints them on its
own in response to actions - minting is driven by the
:class:`~robot.learning.teaching_controller.TeachingController` (Phase 8), so ambient
(non-teaching) actions carry ``interaction_id=None`` and produce untagged
experiences. The :class:`~robot.services.executor.ActionExecutor` reads
:meth:`InteractionContext.current_metadata` and merges it into each
transition's metadata, so a real action executed during a teaching session is
tagged with the session it belongs to.

Thread-safe via a ``threading.Lock`` - the teaching controller runs on the
event loop while the training cycle may read metadata from a worker, so the
accessors are guarded.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


def _mint_id() -> str:
    """Mint a fresh identifier."""
    return str(uuid.uuid4())


@dataclass(slots=True)
class InteractionContext:
    """Holds the active interaction/teaching-session/episode identifiers.

    The identifiers are ``None`` outside an active span. Each ``begin_*`` mints
    a fresh uuid4 and returns it; the matching ``end_*`` clears it. None of the
    accessors auto-mint - they report the *current* state only, so ambient
    actions stay untagged unless a controller explicitly opened a span.
    """

    interaction_id: str | None = None
    teaching_session_id: str | None = None
    episode_id: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # ------------------------------------------------------------------ spans
    def begin_interaction(self) -> str:
        """Mint and set a new interaction id; returns the minted id."""
        new_id = _mint_id()
        with self._lock:
            self.interaction_id = new_id
        return new_id

    def end_interaction(self) -> None:
        """Clear the active interaction id."""
        with self._lock:
            self.interaction_id = None

    def begin_teaching_session(self) -> str:
        """Mint and set a new teaching-session id; returns the minted id."""
        new_id = _mint_id()
        with self._lock:
            self.teaching_session_id = new_id
        return new_id

    def end_teaching_session(self) -> None:
        """Clear the active teaching-session id."""
        with self._lock:
            self.teaching_session_id = None

    def begin_episode(self) -> str:
        """Mint and set a new episode id; returns the minted id."""
        new_id = _mint_id()
        with self._lock:
            self.episode_id = new_id
        return new_id

    def end_episode(self) -> None:
        """Clear the active episode id."""
        with self._lock:
            self.episode_id = None

    # ------------------------------------------------------------------ reads
    def current_metadata(self) -> dict[str, Any]:
        """Return the non-None identifiers, for merging into transition metadata.

        Keys with a ``None`` value are omitted so ambient (non-teaching)
        actions produce an empty dict and the executor's merge is a no-op.
        """
        with self._lock:
            interaction_id = self.interaction_id
            teaching_session_id = self.teaching_session_id
            episode_id = self.episode_id
        metadata: dict[str, Any] = {}
        if interaction_id is not None:
            metadata["interaction_id"] = interaction_id
        if teaching_session_id is not None:
            metadata["teaching_session_id"] = teaching_session_id
        if episode_id is not None:
            metadata["episode_id"] = episode_id
        return metadata

    def reset(self) -> None:
        """Clear all active identifiers (e.g. on shutdown or session abort)."""
        with self._lock:
            self.interaction_id = None
            self.teaching_session_id = None
            self.episode_id = None


__all__ = ["InteractionContext"]
