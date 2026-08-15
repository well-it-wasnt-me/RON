"""Activation functions and their derivatives.

Each activation is a pair of functions: the forward pass and the
derivative (used in backpropagation).  All operate on :class:`Tensor`
instances and return :class:`Tensor` instances.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from robot.learning.tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """Rectified Linear Unit: ``max(0, x)``."""
    return Tensor(np.maximum(0.0, x.data))


def relu_derivative(x: Tensor) -> Tensor:
    """Derivative of ReLU: 1 where x > 0, else 0."""
    return Tensor((x.data > 0).astype(np.float64))


def sigmoid(x: Tensor) -> Tensor:
    """Logistic sigmoid: ``1 / (1 + exp(-x))``."""
    # Numerically stable: clamp input to avoid overflow
    clipped = np.clip(x.data, -500.0, 500.0)
    return Tensor(1.0 / (1.0 + np.exp(-clipped)))


def sigmoid_derivative(x: Tensor) -> Tensor:
    """Derivative of sigmoid given the *output* of sigmoid.

    ``sigma'(x) = sigma(x) * (1 - sigma(x))``

    Pass the **output** of ``sigmoid``, not the raw input.
    """
    s = x.data
    return Tensor(s * (1.0 - s))


def tanh(x: Tensor) -> Tensor:
    """Hyperbolic tangent."""
    return Tensor(np.tanh(x.data))


def tanh_derivative(x: Tensor) -> Tensor:
    """Derivative of tanh given the *output* of tanh.

    ``tanh'(x) = 1 - tanh(x)^2``

    Pass the **output** of ``tanh``, not the raw input.
    """
    t = x.data
    return Tensor(1.0 - t * t)


def linear(x: Tensor) -> Tensor:
    """Identity activation (no-op)."""
    return x


def linear_derivative(x: Tensor) -> Tensor:
    """Derivative of identity: always 1."""
    return Tensor(np.ones_like(x.data))


def softmax(x: Tensor) -> Tensor:
    """Numerically stable softmax along the last axis."""
    shifted = x.data - np.max(x.data, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return Tensor(exp_vals / np.sum(exp_vals, axis=-1, keepdims=True))


def softmax_derivative(softmax_output: Tensor) -> Tensor:
    """Derivative of softmax (Jacobian diagonal approximation for cross-entropy).

    When used with cross-entropy loss, the combined derivative simplifies
    to ``softmax_output - target``, so this returns the softmax output
    itself for the caller to combine.
    """
    return softmax_output


# Registry of named activations
ACTIVATIONS: dict[str, tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]] = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "linear": (linear, linear_derivative),
    "softmax": (softmax, softmax_derivative),
}


def get_activation(name: str) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return (forward, derivative) for a named activation."""
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; available: {sorted(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]


__all__ = [
    "ACTIVATIONS",
    "Tensor",
    "get_activation",
    "linear",
    "linear_derivative",
    "relu",
    "relu_derivative",
    "sigmoid",
    "sigmoid_derivative",
    "softmax",
    "softmax_derivative",
    "tanh",
    "tanh_derivative",
]
