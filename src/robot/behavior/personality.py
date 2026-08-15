"""Lightweight personality engine.

Personality is a frozen bag of traits in [0, 1]. The values are used to weight
the probabilities of idle actions (curiosity -> more glances, shyness -> more
looking away, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from robot.config import PersonalityConfig


class PersonalityTrait(str, Enum):
    CURIOSITY = "curiosity"
    ENERGY = "energy"
    SHYNESS = "shyness"
    FRIENDLINESS = "friendliness"
    PLAYFULNESS = "playfulness"


@dataclass(slots=True, frozen=True)
class Personality:
    """Immutable bag of personality traits in [0, 1]."""

    curiosity: float
    energy: float
    shyness: float
    friendliness: float
    playfulness: float

    @classmethod
    def from_config(cls, config: PersonalityConfig) -> Personality:
        return cls(
            curiosity=config.curiosity,
            energy=config.energy,
            shyness=config.shyness,
            friendliness=config.friendliness,
            playfulness=config.playfulness,
        )

    def value(self, trait: PersonalityTrait) -> float:
        return float(getattr(self, trait.value))

    def with_trait(self, trait: PersonalityTrait, value: float) -> Personality:
        clamped = max(0.0, min(1.0, value))
        return Personality(
            curiosity=clamped if trait is PersonalityTrait.CURIOSITY else self.curiosity,
            energy=clamped if trait is PersonalityTrait.ENERGY else self.energy,
            shyness=clamped if trait is PersonalityTrait.SHYNESS else self.shyness,
            friendliness=clamped if trait is PersonalityTrait.FRIENDLINESS else self.friendliness,
            playfulness=clamped if trait is PersonalityTrait.PLAYFULNESS else self.playfulness,
        )


__all__ = ["Personality", "PersonalityTrait"]
