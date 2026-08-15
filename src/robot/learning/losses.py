"""Loss functions and their derivatives for network training.

Each loss function returns a scalar :class:`Tensor` and has a
corresponding derivative function that returns the gradient with
respect to the prediction.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from robot.learning.tensor import Tensor


def mse_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error loss.

    ``L = mean((predicted - target)^2)``
    """
    diff = predicted - target
    return (diff * diff).mean()


def mse_derivative(predicted: Tensor, target: Tensor) -> Tensor:
    """Gradient of MSE loss w.r.t. ``predicted``.

    ``dL/dpredicted = 2 * (predicted - target) / n``
    """
    n = predicted.size
    return (predicted - target) * (2.0 / n)


def cross_entropy_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """Cross-entropy loss for classification.

    ``predicted`` should be softmax probabilities (batch_size, classes).
    ``target`` should be one-hot encoded (batch_size, classes).

    ``L = -mean(sum(target * log(predicted + eps)))``
    """
    eps = 1e-12
    clipped = np.clip(predicted.data, eps, 1.0 - eps)
    log_probs = Tensor(np.log(clipped))
    elementwise = target * log_probs
    return -(elementwise.sum() / predicted.shape[0])


def cross_entropy_derivative(predicted: Tensor, target: Tensor) -> Tensor:
    """Gradient of cross-entropy loss w.r.t. ``predicted``.

    When used with softmax output, the combined gradient is:

    ``dL/dz = predicted - target``

    This returns that simplified form.
    """
    return (predicted - target) * (1.0 / predicted.shape[0])


# Registry of named losses
LOSSES: dict[str, tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor, Tensor], Tensor]]] = {
    "mse": (mse_loss, mse_derivative),
    "cross_entropy": (cross_entropy_loss, cross_entropy_derivative),
}


def get_loss(
    name: str,
) -> tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor, Tensor], Tensor]]:
    """Return (loss_fn, derivative_fn) for a named loss."""
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; available: {sorted(LOSSES.keys())}")
    return LOSSES[name]


__all__ = [
    "LOSSES",
    "cross_entropy_derivative",
    "cross_entropy_loss",
    "get_loss",
    "mse_derivative",
    "mse_loss",
]
