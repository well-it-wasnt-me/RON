"""Local preference learning from experience observations.

This module extends DeskBot's existing :class:`PreferenceTracker` with a
learning layer that:

1. Observes recurring patterns in experiences (rewards, actions, states).
2. Builds confidence from repeated observations.
3. Decays confidence when preferences are not reinforced.
4. Persists preferences across restarts via the existing
   :class:`PreferenceStore` backends.

The preference learner does not infer sensitive personal attributes. It only
learns preferences relevant to DeskBot's operation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypedDict, cast

from robot.ai.preferences import (
    InMemoryPreferenceStore,
    Preference,
    PreferenceStore,
)
from robot.logging import get_logger

_log = get_logger("learning.preference_learner")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PREFERENCE_CATEGORIES = frozenset(
    {
        "interaction_style",
        "preferred_action",
        "emotional_response",
        "interaction_time",
        "face_preference",
        "volume_preference",
    }
)

_EXPLICIT_BOOST = 0.3
_INFERRED_BOOST = 0.15

_MIN_OBSERVATIONS = 3
_CONFIDENCE_THRESHOLD = 0.5
_DAILY_DECAY_RATE = 0.02
_MIN_CONFIDENCE = 0.1


# ---------------------------------------------------------------------------
# Persistence row types
# ---------------------------------------------------------------------------


class _LearnedPreferenceRow(TypedDict):
    """Typed representation of a persisted learned-preference row."""

    key: str
    value: str
    confidence: float
    source: str
    observation_count: int
    total_reward: float
    first_observed: datetime
    last_observed: datetime


# ---------------------------------------------------------------------------
# Pattern observation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PatternObservation:
    """A single observation of a recurring pattern."""

    category: str
    value: str
    timestamp: datetime
    reward: float = 0.0
    source: str = "behavioral"


# ---------------------------------------------------------------------------
# Learned preference
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LearnedPreference:
    """A preference learned from repeated observations."""

    key: str
    category: str
    value: str
    confidence: float = 0.2
    observation_count: int = 0
    first_observed: datetime = field(
        default_factory=lambda: datetime.now(tz=UTC)
    )
    last_observed: datetime = field(
        default_factory=lambda: datetime.now(tz=UTC)
    )
    total_reward: float = 0.0
    source: str = "behavioral"

    @property
    def avg_reward(self) -> float:
        """Average reward across all observations."""
        return self.total_reward / max(self.observation_count, 1)

    def to_preference(self) -> Preference:
        """Convert this learned preference to a basic Preference."""
        return Preference(
            key=self.key,
            value=self.value,
            confidence=self.confidence,
            source=self.source,
            updated_at=self.last_observed,
        )


# ---------------------------------------------------------------------------
# Preference learner
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PreferenceLearner:
    """Learn preferences from experience observations."""

    store: PreferenceStore = field(
        default_factory=InMemoryPreferenceStore
    )
    decay_days: float = 1.0
    min_confidence: float = _MIN_CONFIDENCE

    _patterns: dict[str, LearnedPreference] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    _observation_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int),
        init=False,
        repr=False,
    )

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    def observe(
        self,
        category: str,
        value: str,
        reward: float = 0.0,
        source: str = "behavioral",
        timestamp: datetime | None = None,
    ) -> LearnedPreference | None:
        """Observe a pattern and update its confidence."""

        if timestamp is None:
            timestamp = datetime.now(tz=UTC)

        key = f"{category}:{value}"

        self._observation_counts[key] += 1

        existing = self._patterns.get(key)

        if existing is not None:
            boost = (
                _EXPLICIT_BOOST
                if source == "explicit"
                else _INFERRED_BOOST
            )

            existing.confidence = min(
                1.0,
                existing.confidence + boost,
            )
            existing.observation_count += 1
            existing.last_observed = timestamp
            existing.total_reward += reward

            if (
                source == "explicit"
                and existing.source == "behavioral"
            ):
                existing.source = "explicit"

            pattern = existing

        else:
            initial_confidence = (
                0.5 if source == "explicit" else 0.2
            )

            pattern = LearnedPreference(
                key=key,
                category=category,
                value=value,
                confidence=initial_confidence,
                observation_count=1,
                first_observed=timestamp,
                last_observed=timestamp,
                total_reward=reward,
                source=source,
            )

            self._patterns[key] = pattern

        if (
            pattern.confidence >= _CONFIDENCE_THRESHOLD
            or pattern.observation_count >= _MIN_OBSERVATIONS
        ):
            self._persist_pattern(pattern)
            return pattern

        return None

    def _persist_pattern(
        self,
        pattern: LearnedPreference,
    ) -> None:
        """Persist a learned preference."""

        from robot.ai.preferences import SqlitePreferenceStore

        if isinstance(self.store, SqlitePreferenceStore):
            try:
                self.store.save_learned(
                    key=pattern.key,
                    value=pattern.value,
                    confidence=pattern.confidence,
                    source=pattern.source,
                    updated_at=pattern.last_observed,
                    observation_count=pattern.observation_count,
                    total_reward=pattern.total_reward,
                    first_observed=pattern.first_observed,
                    last_observed=pattern.last_observed,
                )
                return
            except Exception:
                _log.warning(
                    "preference_learner.persist_fallback",
                    reason="extended_save_failed",
                )

        self.store.save(pattern.to_preference())

    # ------------------------------------------------------------------
    # Reward observations
    # ------------------------------------------------------------------

    def observe_from_reward(
        self,
        action_name: str,
        reward: float,
        state_context: dict[str, Any] | None = None,
    ) -> LearnedPreference | None:
        """Observe an action/reward pair as a preference signal."""

        # Keep this argument intentionally accepted for callers that provide
        # contextual information. The current learner does not persist it.
        _ = state_context

        if reward > 0.5:
            return self.observe(
                category="preferred_action",
                value=action_name,
                reward=reward,
                source="behavioral",
            )

        if reward < -0.5:
            return self.observe(
                category="preferred_action",
                value=f"not_{action_name}",
                reward=reward,
                source="behavioral",
            )

        return None

    def observe_interaction_style(
        self,
        style: str,
        reward: float = 0.0,
        source: str = "behavioral",
    ) -> LearnedPreference | None:
        """Observe an interaction-style preference."""
        return self.observe(
            category="interaction_style",
            value=style,
            reward=reward,
            source=source,
        )

    def observe_emotional_response(
        self,
        emotion: str,
        reward: float = 0.0,
    ) -> LearnedPreference | None:
        """Observe a preferred emotional response."""
        return self.observe(
            category="emotional_response",
            value=emotion,
            reward=reward,
        )

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def apply_decay(
        self,
        now: datetime | None = None,
    ) -> list[str]:
        """Decay confidence based on time since last observation."""

        if self.decay_days <= 0:
            return []

        if now is None:
            now = datetime.now(tz=UTC)

        daily_decay_rate = math.log(2) / self.decay_days

        decayed: list[str] = []

        for key, pattern in self._patterns.items():
            days_since = (
                now - pattern.last_observed
            ).total_seconds() / 86400.0

            if days_since <= 0:
                continue

            confidence_above_min = (
                pattern.confidence - self.min_confidence
            )

            if confidence_above_min <= 0:
                continue

            new_confidence_above_min = (
                confidence_above_min
                * math.exp(-daily_decay_rate * days_since)
            )

            new_confidence = (
                self.min_confidence
                + new_confidence_above_min
            )

            if new_confidence < pattern.confidence:
                pattern.confidence = new_confidence
                decayed.append(key)

                if pattern.observation_count >= _MIN_OBSERVATIONS:
                    self._persist_pattern(pattern)

        removed: list[str] = []

        for key, pattern in self._patterns.items():
            if (
                pattern.confidence <= self.min_confidence
                and pattern.observation_count < _MIN_OBSERVATIONS
            ):
                removed.append(key)

        for key in removed:
            del self._patterns[key]
            self.store.delete(key)

        return decayed

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_preference(
        self,
        category: str,
        value: str | None = None,
    ) -> LearnedPreference | None:
        """Return a preference for a category/value."""

        if value is not None:
            return self._patterns.get(
                f"{category}:{value}"
            )

        best: LearnedPreference | None = None

        for pattern in self._patterns.values():
            if pattern.category != category:
                continue

            if (
                best is None
                or pattern.confidence > best.confidence
            ):
                best = pattern

        return best

    def get_all_preferences(
        self,
        category: str | None = None,
    ) -> list[LearnedPreference]:
        """Return all tracked preferences."""

        prefs = list(self._patterns.values())

        if category is not None:
            prefs = [
                preference
                for preference in prefs
                if preference.category == category
            ]

        prefs.sort(
            key=lambda preference: preference.confidence,
            reverse=True,
        )

        return prefs

    def get_learned_preferences(
        self,
        min_confidence: float = _CONFIDENCE_THRESHOLD,
    ) -> list[LearnedPreference]:
        """Return preferences meeting the confidence threshold."""

        return [
            preference
            for preference in self._patterns.values()
            if preference.confidence >= min_confidence
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_learned_row(
        raw: dict[str, Any],
    ) -> _LearnedPreferenceRow:
        """Convert an untyped persistence row into a typed row.

        The persistence layer predates the learner and currently exposes
        generic dictionaries. Keeping the boundary conversion here prevents
        ``object`` values from leaking into the strongly typed learner.
        """

        key = raw.get("key")
        value = raw.get("value")
        confidence = raw.get("confidence")
        source = raw.get("source")
        observation_count = raw.get("observation_count")
        total_reward = raw.get("total_reward")
        first_observed = raw.get("first_observed")
        last_observed = raw.get("last_observed")

        if not isinstance(key, str):
            raise TypeError("learned preference key must be a string")

        if not isinstance(value, str):
            raise TypeError("learned preference value must be a string")

        if not isinstance(confidence, (int, float)):
            raise TypeError(
                "learned preference confidence must be numeric"
            )

        if not isinstance(source, str):
            raise TypeError(
                "learned preference source must be a string"
            )

        if not isinstance(observation_count, int):
            raise TypeError(
                "learned preference observation_count must be an int"
            )

        if not isinstance(total_reward, (int, float)):
            raise TypeError(
                "learned preference total_reward must be numeric"
            )

        if not isinstance(first_observed, datetime):
            raise TypeError(
                "learned preference first_observed must be datetime"
            )

        if not isinstance(last_observed, datetime):
            raise TypeError(
                "learned preference last_observed must be datetime"
            )

        return {
            "key": key,
            "value": value,
            "confidence": float(confidence),
            "source": source,
            "observation_count": observation_count,
            "total_reward": float(total_reward),
            "first_observed": first_observed,
            "last_observed": last_observed,
        }

    def load_from_store(self) -> None:
        """Load previously persisted preferences."""

        from robot.ai.preferences import SqlitePreferenceStore

        if isinstance(self.store, SqlitePreferenceStore):
            try:
                raw_rows = self.store.load_learned_all()

                # The current PreferenceStore implementation returns generic
                # dictionaries. Validate each row at this boundary so mypy
                # and runtime consumers both have concrete types.
                for raw_row in raw_rows:
                    row = self._coerce_learned_row(
                        cast("dict[str, Any]", raw_row)
                    )

                    key = row["key"]

                    if ":" in key:
                        category, value = key.split(":", 1)
                    else:
                        category = "unknown"
                        value = row["value"]

                    self._patterns[key] = LearnedPreference(
                        key=key,
                        category=category,
                        value=value,
                        confidence=row["confidence"],
                        observation_count=row["observation_count"],
                        first_observed=row["first_observed"],
                        last_observed=row["last_observed"],
                        total_reward=row["total_reward"],
                        source=row["source"],
                    )

                _log.info(
                    "preference_learner.loaded",
                    count=len(self._patterns),
                    format="extended",
                )
                return

            except Exception:
                _log.warning(
                    "preference_learner.load_fallback",
                    reason="extended_load_failed",
                )

        prefs = self.store.load_all()

        for pref in prefs:
            key = pref.key

            if ":" in key:
                category, value = key.split(":", 1)
            else:
                category = "unknown"
                value = pref.value

            self._patterns[key] = LearnedPreference(
                key=key,
                category=category,
                value=value,
                confidence=pref.confidence,
                observation_count=max(
                    1,
                    int(pref.confidence * 10),
                ),
                first_observed=pref.updated_at,
                last_observed=pref.updated_at,
                source=pref.source,
            )

        _log.info(
            "preference_learner.loaded",
            count=len(self._patterns),
            format="basic",
        )

    def clear(self) -> None:
        """Clear all learned preferences from memory and persistence."""

        self._patterns.clear()
        self._observation_counts.clear()

        for pref in self.store.load_all():
            self.store.delete(pref.key)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def preference_count(self) -> int:
        """Number of preferences meeting the confidence threshold."""
        return len(self.get_learned_preferences())

    @property
    def total_patterns(self) -> int:
        """Total number of tracked patterns."""
        return len(self._patterns)


__all__ = [
    "LearnedPreference",
    "PatternObservation",
    "PreferenceLearner",
]
