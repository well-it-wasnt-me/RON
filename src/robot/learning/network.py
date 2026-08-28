"""Neural network model and training loop.

:class:`Network` is the top-level container that owns the layers,
manages forward/backward passes, and provides training utilities.
:class:`MLP` is a convenience factory that builds a standard
multi-layer perceptron.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Protocol, runtime_checkable

from robot.learning.layers import DenseLayer
from robot.learning.losses import (
    cross_entropy_derivative,
    cross_entropy_loss,
    mse_derivative,
    mse_loss,
)
from robot.learning.tensor import Tensor


@runtime_checkable
class Optimizer(Protocol):
    """Protocol for optimisers used by :class:`Network`."""

    def step(self, layers: list[DenseLayer]) -> None: ...


class Network:
    """A feed-forward neural network composed of :class:`DenseLayer` layers.

    The network is constructed with a list of layers and provides
    ``forward``, ``backward``, and ``train_step`` methods.  Model
    parameters can be saved to and loaded from JSON files.
    """

    def __init__(self, layers: list[DenseLayer]) -> None:
        self.layers = layers

    # ------------------------------------------------------------------ forward
    def forward(self, x: Tensor) -> Tensor:
        """Run input through all layers and return the output."""
        current = x
        for layer in self.layers:
            current = layer.forward(current)
        return current

    # ------------------------------------------------------------------ backward
    def backward(self, grad: Tensor) -> Tensor:
        """Back-propagate the loss gradient through all layers.

        Layers are updated in reverse order.  After calling
        ``backward``, each layer's ``weight_grad`` and ``bias_grad``
        fields contain the computed gradients.
        """
        current = grad
        for layer in reversed(self.layers):
            current = layer.backward(current)
        return current

    # ------------------------------------------------------------------ training
    def train_step(
        self,
        x: Tensor,
        target: Tensor,
        loss_fn: str = "mse",
        optimizer: Optimizer | None = None,
    ) -> tuple[float, Tensor]:
        """Run one training step: forward, loss, backward, optimise.

        Parameters
        ----------
        x:
            Input batch.  Shape ``(batch, in_features)``.
        target:
            Target batch.  Shape ``(batch, out_features)``.
        loss_fn:
            Loss function name (``"mse"`` or ``"cross_entropy"``).
        optimizer:
            Optimiser to apply gradients.  If ``None``, gradients are
            computed but no update is applied.

        Returns
        -------
        tuple[float, Tensor]
            ``(loss_value, prediction)`` where ``loss_value`` is a Python
            float and ``prediction`` is the network output.
        """
        prediction = self.forward(x)

        if loss_fn == "mse":
            loss = mse_loss(prediction, target)
            grad = mse_derivative(prediction, target)
        elif loss_fn == "cross_entropy":
            loss = cross_entropy_loss(prediction, target)
            grad = cross_entropy_derivative(prediction, target)
        else:
            raise ValueError(f"unknown loss {loss_fn!r}")

        self.backward(grad)

        # Check for NaN/inf in gradients — skip the step if contaminated
        # to prevent poisoning the weights.
        if optimizer is not None:
            for layer in self.layers:
                if not np.all(np.isfinite(layer.weight_grad.data)):
                    _log_nan = __import__("robot.logging", fromlist=["get_logger"]).get_logger("learning.network")
                    _log_nan.warning("network.nan_gradient_detected", loss=loss.item())
                    return loss.item(), prediction

            optimizer.step(self.layers)

        return loss.item(), prediction

    # ------------------------------------------------------------------ predict
    def predict(self, x: Tensor) -> Tensor:
        """Forward pass for inference.

        This calls :meth:`forward`, which does populate each layer's
        internal cache (``_input``, ``_pre_activation``, ``_output``).
        Those caches are overwritten on the next ``train_step`` call, so
        calling ``predict`` between training steps does not produce
        incorrect gradients.

        However, ``predict`` is **not** side-effect-free: the layer caches
        are set. Do not rely on them being empty after this call.
        """
        return self.forward(x)

    # ------------------------------------------------------------------ state
    def get_state(self) -> dict[str, object]:
        """Return a JSON-serialisable state dict for checkpointing."""
        return {
            "layers": [layer.get_state() for layer in self.layers],
        }

    def save(self, path: str | Path) -> None:
        """Save model state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.get_state()
        path.write_text(json.dumps(state, indent=2))

    @classmethod
    def from_state(cls, state: dict[str, object]) -> Network:
        """Reconstruct a network from a state dict."""
        raw_layers = state["layers"]
        assert isinstance(raw_layers, list)
        layers = [DenseLayer.from_state(ls) for ls in raw_layers]
        return cls(layers)

    @classmethod
    def load(cls, path: str | Path) -> Network:
        """Load model state from a JSON file."""
        path = Path(path)
        state = json.loads(path.read_text())
        return cls.from_state(state)

    # ------------------------------------------------------------------ info
    def param_count(self) -> int:
        """Total number of trainable parameters."""
        total = 0
        for layer in self.layers:
            total += layer.weights.size + layer.biases.size
        return total

    def __repr__(self) -> str:
        layer_descs = " -> ".join(
            f"({layer.in_features}->{layer.out_features}, {layer.activation_name})"
            for layer in self.layers
        )
        return f"Network({layer_descs}, params={self.param_count()})"


class MLP:
    """Convenience factory for a configurable multi-layer perceptron.

    Builds a network of the form::

        input -> Dense -> ReLU -> Dense -> ReLU -> ... -> Dense -> output_activation

    Parameters
    ----------
    input_size:
        Number of input features.
    hidden_sizes:
        List of hidden layer sizes.
    output_size:
        Number of output features.
    activation:
        Activation for hidden layers.  Default ``"relu"``.
    output_activation:
        Activation for the output layer.  Default ``"linear"``.
    weight_init:
        Weight initialisation strategy.
    seed:
        Random seed for reproducible initialisation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: str = "relu",
        output_activation: str = "linear",
        weight_init: str = "he",
        seed: int | None = None,
    ) -> None:
        # Use a seeded RNG to produce per-layer seeds so each layer
        # gets different but reproducible weights.
        master_rng = np.random.default_rng(seed)
        layer_seeds = [int(master_rng.integers(0, 2**31)) for _ in range(len(hidden_sizes) + 1)]

        layers: list[DenseLayer] = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(
                DenseLayer(
                    in_features=prev_size,
                    out_features=hidden_size,
                    activation=activation,
                    weight_init=weight_init,
                    seed=layer_seeds[i],
                )
            )
            prev_size = hidden_size

        # Output layer
        layers.append(
            DenseLayer(
                in_features=prev_size,
                out_features=output_size,
                activation=output_activation,
                weight_init=weight_init,
                seed=layer_seeds[-1],
            )
        )

        self.network = Network(layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network.forward(x)

    def train_step(
        self,
        x: Tensor,
        target: Tensor,
        loss_fn: str = "mse",
        optimizer: Optimizer | None = None,
    ) -> tuple[float, Tensor]:
        return self.network.train_step(x, target, loss_fn, optimizer)

    def predict(self, x: Tensor) -> Tensor:
        return self.network.predict(x)

    def save(self, path: str | Path) -> None:
        self.network.save(path)

    @classmethod
    def load(cls, path: str | Path) -> MLP:
        """Load an MLP from a saved network file.

        Note: the loaded object is an MLP wrapping a Network; hidden
        layer sizes are inferred from the saved layer dimensions.
        """
        network = Network.load(path)
        mlp = cls.__new__(cls)
        mlp.network = network
        return mlp


__all__ = ["MLP", "Network", "Optimizer"]
