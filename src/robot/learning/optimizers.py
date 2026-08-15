"""Optimizers for neural network training.

Each optimizer holds a reference to the layers it updates and applies
gradient-based parameter updates.  Optimizers are decoupled from the
network so they can be swapped without changing the architecture.
"""

from __future__ import annotations

import numpy as np

from robot.learning.layers import DenseLayer
from robot.learning.tensor import Tensor


class SGD:
    """Stochastic Gradient Descent with optional momentum.

    Parameters
    ----------
    learning_rate:
        Step size for parameter updates.
    momentum:
        Momentum coefficient (0.0 = no momentum).
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._velocity_w: list[np.ndarray | None] = []
        self._velocity_b: list[np.ndarray | None] = []
        self._initialised = False

    def step(self, layers: list[DenseLayer]) -> None:
        """Apply one optimisation step to all layers."""
        if not self._initialised:
            self._velocity_w = [None] * len(layers)
            self._velocity_b = [None] * len(layers)
            self._initialised = True

        for i, layer in enumerate(layers):
            if self.momentum > 0.0:
                vw = self._velocity_w[i]
                vb = self._velocity_b[i]
                if vw is None or vb is None:
                    vw = np.zeros_like(layer.weights.data)
                    vb = np.zeros_like(layer.biases.data)

                vw = self.momentum * vw + layer.weight_grad.data
                vb = self.momentum * vb + layer.bias_grad.data
                self._velocity_w[i] = vw
                self._velocity_b[i] = vb

                layer.weights = Tensor(layer.weights.data - self.learning_rate * vw)
                layer.biases = Tensor(layer.biases.data - self.learning_rate * vb)
            else:
                layer.weights = Tensor(
                    layer.weights.data - self.learning_rate * layer.weight_grad.data
                )
                layer.biases = Tensor(layer.biases.data - self.learning_rate * layer.bias_grad.data)


class Adam:
    """Adam optimiser.

    Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.

    Parameters
    ----------
    learning_rate:
        Step size.
    beta1:
        Exponential decay rate for the first moment estimate.
    beta2:
        Exponential decay rate for the second moment estimate.
    eps:
        Small constant for numerical stability.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._t: int = 0
        self._m_w: list[np.ndarray | None] = []
        self._v_w: list[np.ndarray | None] = []
        self._m_b: list[np.ndarray | None] = []
        self._v_b: list[np.ndarray | None] = []
        self._initialised = False

    def step(self, layers: list[DenseLayer]) -> None:
        """Apply one Adam update step."""
        if not self._initialised:
            n = len(layers)
            self._m_w = [None] * n
            self._v_w = [None] * n
            self._m_b = [None] * n
            self._v_b = [None] * n
            self._initialised = True

        self._t += 1

        for i, layer in enumerate(layers):
            m_w = self._m_w[i]
            v_w = self._v_w[i]
            m_b = self._m_b[i]
            v_b = self._v_b[i]
            if m_w is None or v_w is None or m_b is None or v_b is None:
                m_w = np.zeros_like(layer.weights.data)
                v_w = np.zeros_like(layer.weights.data)
                m_b = np.zeros_like(layer.biases.data)
                v_b = np.zeros_like(layer.biases.data)

            # First moment
            m_w = self.beta1 * m_w + (1.0 - self.beta1) * layer.weight_grad.data
            m_b = self.beta1 * m_b + (1.0 - self.beta1) * layer.bias_grad.data

            # Second moment
            v_w = self.beta2 * v_w + (1.0 - self.beta2) * layer.weight_grad.data**2
            v_b = self.beta2 * v_b + (1.0 - self.beta2) * layer.bias_grad.data**2

            self._m_w[i] = m_w
            self._v_w[i] = v_w
            self._m_b[i] = m_b
            self._v_b[i] = v_b

            # Bias correction
            m_hat_w = m_w / (1.0 - self.beta1**self._t)
            v_hat_w = v_w / (1.0 - self.beta2**self._t)
            m_hat_b = m_b / (1.0 - self.beta1**self._t)
            v_hat_b = v_b / (1.0 - self.beta2**self._t)

            # Update
            layer.weights = Tensor(
                layer.weights.data - self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
            )
            layer.biases = Tensor(
                layer.biases.data - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
            )


__all__ = ["SGD", "Adam"]
