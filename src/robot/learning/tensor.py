"""Lightweight tensor wrapper around NumPy arrays.

Provides a thin abstraction so the rest of the learning code never
imports NumPy directly - making it easy to swap backends or add
device-agnostic helpers later.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class Tensor:
    """A named wrapper around a NumPy ndarray."""

    # Explicitly mark Tensor as unhashable because it is mutable.
    # This suppresses ruff PLW1641 and makes the unhashable contract explicit.
    __hash__ = None  # type: ignore[assignment]

    __slots__ = ("_data",)

    _data: np.ndarray

    def __init__(
        self, data: np.ndarray | list[float] | list[list[float]] | Tensor | np.floating
    ) -> None:
        if isinstance(data, Tensor):
            self._data = data._data.astype(np.float64)
        elif isinstance(data, np.ndarray):
            self._data = data.astype(np.float64)
        elif isinstance(data, np.floating):
            self._data = np.array(data, dtype=np.float64)
        else:
            self._data = np.array(data, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def size(self) -> int:
        return int(self._data.size)

    @property
    def ndim(self) -> int:
        return int(self._data.ndim)

    @property
    def data(self) -> np.ndarray:
        """Raw NumPy array (read-only view)."""
        return self._data

    @classmethod
    def zeros(cls, *shape: int) -> Tensor:
        return cls(np.zeros(shape, dtype=np.float64))

    @classmethod
    def ones(cls, *shape: int) -> Tensor:
        return cls(np.ones(shape, dtype=np.float64))

    @classmethod
    def randn(cls, *shape: int, seed: int | None = None) -> Tensor:
        rng = np.random.default_rng(seed)
        return cls(rng.standard_normal(shape, dtype=np.float64))

    @classmethod
    def uniform(cls, low: float, high: float, *shape: int, seed: int | None = None) -> Tensor:
        rng = np.random.default_rng(seed)
        return cls(rng.uniform(low, high, shape).astype(np.float64))

    @classmethod
    def from_row(cls, data: list[float]) -> Tensor:
        """Create a 1-D tensor from a Python list of floats."""
        return cls(np.array(data, dtype=np.float64))

    def __add__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(self._data + other._data)
        return Tensor(self._data + float(other))

    def __radd__(self, other: float | int) -> Tensor:
        return Tensor(float(other) + self._data)

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(self._data - other._data)
        return Tensor(self._data - float(other))

    def __rsub__(self, other: float | int) -> Tensor:
        return Tensor(float(other) - self._data)

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(self._data * other._data)
        return Tensor(self._data * float(other))

    def __rmul__(self, other: float | int) -> Tensor:
        return Tensor(float(other) * self._data)

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, Tensor):
            with np.errstate(divide="ignore", invalid="ignore"):
                return Tensor(self._data / other._data)
        with np.errstate(divide="ignore", invalid="ignore"):
            return Tensor(self._data / float(other))

    def __neg__(self) -> Tensor:
        return Tensor(-self._data)

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor(self._data @ other._data)

    def sum(self, axis: int | None = None) -> Tensor:
        result = self._data.sum(axis=axis)
        return Tensor(np.array(result, dtype=np.float64))

    def mean(self, axis: int | None = None) -> Tensor:
        return Tensor(self._data.mean(axis=axis))

    def reshape(self, *shape: int) -> Tensor:
        return Tensor(self._data.reshape(shape))

    def transpose(self) -> Tensor:
        """Matrix transpose (2-D only)."""
        return Tensor(self._data.T)

    def T(self) -> Tensor:  # noqa: N802
        """Matrix transpose (method form, matching the public API)."""
        return self.transpose()

    def clip(self, low: float, high: float) -> Tensor:
        return Tensor(np.clip(self._data, low, high))

    def flatten(self) -> Tensor:
        return Tensor(self._data.ravel())

    def item(self) -> float:
        """Return the tensor as a Python float (must be scalar)."""
        return float(self._data.item())

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Tensor]:
        for row in self._data:
            yield Tensor(row)

    def __getitem__(self, idx: int | slice) -> Tensor:
        return Tensor(self._data[idx])

    def __setitem__(self, idx: int | slice, value: Tensor | float) -> None:
        if isinstance(value, Tensor):
            self._data[idx] = value._data
        else:
            self._data[idx] = float(value)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self._data})"

    def __eq__(self, other: object) -> bool:
        """Value-based equality for testing and comparison.

        Compares the underlying arrays elementwise using ``np.array_equal``.
        Tensors of different shapes are never equal (returns ``False``).
        Comparing a Tensor with a non-Tensor returns ``NotImplemented`` so
        Python falls back to identity comparison.

        Because Tensor is mutable (has ``__setitem__``), ``__hash__`` is
        intentionally **not** defined. Python therefore treats instances as
        unhashable (``__hash__ = None``), preventing accidental use as dict
        keys or set members — which would break when the underlying data
        is mutated.
        """
        if not isinstance(other, Tensor):
            return NotImplemented
        return np.array_equal(self._data, other._data)


__all__ = ["Tensor"]
