"""Neural network layers.

Each layer stores its own weights, biases, and gradients.  The
``forward`` method computes the output; the ``backward`` method
computes gradients and returns the gradient w.r.t. the input.

Gradient convention
-------------------
The weight and bias gradients are averaged over the mini-batch
(divided by ``batch_size``).  This makes the gradient magnitude
independent of batch size, so the same learning rate works
regardless of how many samples are in a batch.

The loss derivative (e.g. ``mse_derivative``) normalises by the
total number of elements ``N = batch_size * output_features``.
Combined with the ``1/batch_size`` averaging in backward, the
effective normalisation is ``1/(batch * features) * 1/batch
= 1/(batch^2 * features)``.  To keep things simple and consistent,
the loss derivative normalises by ``N`` (total elements), and
backward divides by ``batch_size``.  This is the same convention
used by PyTorch with ``reduction='mean'``.

Numerical gradient checks should use the same loss function (e.g.
``mse_loss``) without any additional normalisation, because the
numerical gradient directly computes ``dL/dW`` where ``L`` is the
scalar loss.  Since the analytical gradient includes the ``1/batch``
averaging, the numerical gradient must be divided by ``batch_size``
to match, or equivalently, the analytical gradient must be multiplied
by ``batch_size``.  In practice, the test should compute the numerical
gradient of the *per-sample* loss (i.e. divide the total MSE by
``batch_size`` instead of ``N``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np

from robot.learning.activations import get_activation
from robot.learning.tensor import Tensor


class DenseLayer:
    """A fully-connected (dense) layer.

    Parameters
    ----------
    in_features:
        Number of input features.
    out_features:
        Number of output features.
    activation:
        Name of the activation function (``"relu"``, ``"sigmoid"``,
        ``"tanh"``, ``"linear"``, ``"softmax"``).  Default is
        ``"linear"`` (no activation).
    weight_init:
        Weight initialisation strategy.  ``"he"`` for He (Kaiming)
        initialisation (good for ReLU), ``"xavier"`` for Xavier/Glorot
        (good for sigmoid/tanh), ``"normal"`` for small random normal.
    seed:
        Random seed for reproducible initialisation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "linear",
        weight_init: str = "he",
        seed: int | None = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self._forward_fn: Callable[[Tensor], Tensor]
        self._derivative_fn: Callable[[Tensor], Tensor]
        self._forward_fn, self._derivative_fn = get_activation(activation)

        rng = np.random.default_rng(seed)
        if weight_init == "he":
            # He/Kaiming initialisation for ReLU networks
            std = np.sqrt(2.0 / in_features)
            self.weights = Tensor(rng.normal(0, std, (in_features, out_features)))
        elif weight_init == "xavier":
            # Xavier/Glorot initialisation for sigmoid/tanh networks
            std = np.sqrt(2.0 / (in_features + out_features))
            self.weights = Tensor(rng.normal(0, std, (in_features, out_features)))
        else:
            # Small random normal
            self.weights = Tensor(rng.normal(0, 0.01, (in_features, out_features)))

        self.biases = Tensor.zeros(out_features)

        # Gradients (populated during backward pass)
        self.weight_grad: Tensor = Tensor.zeros(in_features, out_features)
        self.bias_grad: Tensor = Tensor.zeros(out_features)

        # Cache for backward pass
        self._input: Tensor | None = None
        self._pre_activation: Tensor | None = None
        self._output: Tensor | None = None

    # ------------------------------------------------------------------ forward
    def forward(self, x: Tensor) -> Tensor:
        """Compute ``activation(x @ weights + biases)``."""
        self._input = x
        # x: (batch, in_features), weights: (in_features, out_features)
        pre = Tensor(x.data @ self.weights.data + self.biases.data)
        self._pre_activation = pre
        output = self._forward_fn(pre)
        self._output = output
        return output

    # ------------------------------------------------------------------ backward
    def backward(self, grad_output: Tensor) -> Tensor:
        """Compute gradients and return gradient w.r.t. input.

        Parameters
        ----------
        grad_output:
            Gradient of the loss w.r.t. the output of this layer.
            Shape: ``(batch, out_features)``.

        Returns
        -------
        Tensor
            Gradient w.r.t. the input of this layer.
            Shape: ``(batch, in_features)``.
        """
        assert self._input is not None, "backward() called before forward()"
        assert self._pre_activation is not None
        assert self._output is not None

        # Activation derivative
        if self.activation_name == "softmax":
            # For softmax + cross-entropy, the combined derivative
            # is (softmax_output - target). The derivative_fn here
            # just returns the output itself, so grad_output is
            # already the combined gradient.
            grad_pre_activation = grad_output
        else:
            # Element-wise activations expose derivatives in terms of the
            # activation output.  This is equivalent for ReLU/linear and
            # required for sigmoid/tanh, whose derivatives are expressed as
            # y * (1 - y) and 1 - y^2 respectively.
            act_deriv = self._derivative_fn(self._output)
            grad_pre_activation = Tensor(grad_output.data * act_deriv.data)

        # grad_output already includes any reduction performed by the loss.
        # Do not average over the batch a second time here.
        self.weight_grad = Tensor(self._input.data.T @ grad_pre_activation.data)
        self.bias_grad = Tensor(grad_pre_activation.data.sum(axis=0))

        # Input gradient: grad_pre_activation @ weights^T
        # (NOT divided by batch_size - propagates per-sample gradients)
        return Tensor(grad_pre_activation.data @ self.weights.data.T)

    # ------------------------------------------------------------------ state
    def get_state(self) -> dict[str, object]:
        """Return a serialisable state dict for checkpointing."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "activation": self.activation_name,
            "weights": self.weights.data.tolist(),
            "biases": self.biases.data.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> DenseLayer:
        """Reconstruct a layer from a state dict."""
        layer = cls.__new__(cls)
        layer.in_features = cast("int", state["in_features"])
        layer.out_features = cast("int", state["out_features"])
        layer.activation_name = cast("str", state["activation"])
        layer._forward_fn, layer._derivative_fn = get_activation(str(state["activation"]))
        layer.weights = Tensor(np.array(state["weights"], dtype=np.float64))
        layer.biases = Tensor(np.array(state["biases"], dtype=np.float64))
        layer.weight_grad = Tensor.zeros(layer.in_features, layer.out_features)
        layer.bias_grad = Tensor.zeros(layer.out_features)
        layer._input = None
        layer._pre_activation = None
        layer._output = None
        return layer

    def __repr__(self) -> str:
        return (
            f"DenseLayer({self.in_features} -> {self.out_features}, "
            f"activation={self.activation_name!r})"
        )


__all__ = ["DenseLayer"]
