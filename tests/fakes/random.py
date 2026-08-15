"""Deterministic random source for tests."""

from __future__ import annotations

from collections.abc import Sequence


class FakeRandom:
    """Returns scripted values, then a constant."""

    def __init__(self, script: Sequence[float] | None = None) -> None:
        self._script = list(script or [])
        self._index = 0

    def _next(self) -> float:
        if self._index < len(self._script):
            value = self._script[self._index]
            self._index += 1
            return value
        return 0.5

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return low + (high - low) * self._next()

    def randint(self, low: int, high: int) -> int:
        return int(low + (high - low) * self._next())

    def choice[T](self, sequence: list[T]) -> T:
        if not sequence:
            raise ValueError("cannot choose from empty sequence")
        return sequence[int(self._next() * len(sequence)) % len(sequence)]

    def random(self) -> float:
        return self._next()
