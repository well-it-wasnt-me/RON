"""Random source abstraction for testable behaviour probabilities."""

from __future__ import annotations

import random
from typing import Protocol


class RandomSource(Protocol):
    """Stateless interface to a random number generator."""

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """A random float in [low, high]."""

    def randint(self, low: int, high: int) -> int:
        """A random integer N such that ``low <= N <= high``."""

    def choice[T](self, sequence: list[T]) -> T:
        """Pick one element of ``sequence``."""

    def random(self) -> float:
        """A random float in [0.0, 1.0)."""


class SystemRandomSource:
    """Production random source backed by :mod:`random`."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return self._rng.uniform(low, high)

    def randint(self, low: int, high: int) -> int:
        return self._rng.randint(low, high)

    def choice[T](self, sequence: list[T]) -> T:
        if not sequence:
            raise ValueError("cannot choose from empty sequence")
        return self._rng.choice(sequence)

    def random(self) -> float:
        return self._rng.random()


__all__ = ["RandomSource", "SystemRandomSource"]
