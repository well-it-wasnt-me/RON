"""Tests for DenseLayer forward, backward, gradient checking, and state."""

import numpy as np

from robot.learning.layers import DenseLayer
from robot.learning.tensor import Tensor


class TestDenseLayerForward:
    def test_output_shape(self) -> None:
        layer = DenseLayer(4, 3, activation="relu", seed=1)
        x = Tensor(np.random.randn(2, 4))
        y = layer.forward(x)
        assert y.shape == (2, 3)

    def test_linear_activation_identity(self) -> None:
        layer = DenseLayer(2, 2, activation="linear", weight_init="normal", seed=42)
        layer.weights = Tensor([[1.0, 0.0], [0.0, 1.0]])
        layer.biases = Tensor([0.0, 0.0])
        x = Tensor([[3.0, 5.0]])
        y = layer.forward(x)
        assert np.allclose(y.data, [[3.0, 5.0]])

    def test_relu_zeros_negative(self) -> None:
        layer = DenseLayer(2, 2, activation="relu", weight_init="normal", seed=42)
        layer.weights = Tensor([[1.0, -1.0], [1.0, -1.0]])
        layer.biases = Tensor([0.0, 0.0])
        x = Tensor([[1.0, 1.0]])
        y = layer.forward(x)
        # Pre-activation: [1+1, -1-1] = [2, -2]
        assert np.allclose(y.data, [[2.0, 0.0]])

    def test_sigmoid_range(self) -> None:
        layer = DenseLayer(2, 2, activation="sigmoid", weight_init="normal", seed=42)
        layer.weights = Tensor([[1.0, 2.0], [3.0, 4.0]])
        layer.biases = Tensor([0.0, 0.0])
        x = Tensor([[1.0, 1.0]])
        y = layer.forward(x)
        # All sigmoid outputs should be in (0, 1)
        assert np.all(y.data > 0)
        assert np.all(y.data < 1)


class TestDenseLayerBackward:
    def test_gradient_shapes(self) -> None:
        layer = DenseLayer(4, 3, activation="relu", seed=1)
        x = Tensor(np.random.randn(2, 4))
        layer.forward(x)
        grad_out = Tensor(np.random.randn(2, 3))
        grad_in = layer.backward(grad_out)
        assert grad_in.shape == (2, 4)
        assert layer.weight_grad.shape == (4, 3)
        assert layer.bias_grad.shape == (3,)

    def test_linear_gradient_correctness(self) -> None:
        """Verify gradients for a linear (identity) layer with known weights."""
        layer = DenseLayer(2, 2, activation="linear", weight_init="normal", seed=42)
        layer.weights = Tensor([[1.0, 2.0], [3.0, 4.0]])
        layer.biases = Tensor([0.0, 0.0])
        x = Tensor([[1.0, 1.0]])
        _ = layer.forward(x)
        grad_out = Tensor([[1.0, 1.0]])
        grad_in = layer.backward(grad_out)
        # Input gradient: grad_out @ weights^T
        expected_grad_in = grad_out.data @ layer.weights.data.T
        assert np.allclose(grad_in.data, expected_grad_in)


class TestDenseLayerNumericalGradient:
    """Numerical gradient checking for DenseLayer.

    The backward pass divides weight/bias gradients by batch_size
    (averaging over the batch).  To compare with numerical gradients,
    we compute the numerical gradient of the total MSE loss and then
    divide by batch_size to match the convention.
    """

    def _numerical_gradient(
        self,
        layer: DenseLayer,
        x: Tensor,
        target: Tensor,
        eps: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute numerical gradients using finite differences on MSE loss."""
        from robot.learning.losses import mse_loss

        batch_size = x.shape[0]
        weight_grads = np.zeros_like(layer.weights.data)
        bias_grads = np.zeros_like(layer.biases.data)

        # Weight gradients
        for i in range(layer.weights.data.shape[0]):
            for j in range(layer.weights.data.shape[1]):
                original = layer.weights.data[i, j]

                layer.weights.data[i, j] = original + eps
                pred_plus = layer.forward(x)
                loss_plus = mse_loss(pred_plus, target).item()

                layer.weights.data[i, j] = original - eps
                pred_minus = layer.forward(x)
                loss_minus = mse_loss(pred_minus, target).item()

                # Divide by batch_size to match backward's convention
                weight_grads[i, j] = (loss_plus - loss_minus) / (2 * eps * batch_size)
                layer.weights.data[i, j] = original

        # Bias gradients
        for j in range(layer.biases.data.shape[0]):
            original = layer.biases.data[j]

            layer.biases.data[j] = original + eps
            pred_plus = layer.forward(x)
            loss_plus = mse_loss(pred_plus, target).item()

            layer.biases.data[j] = original - eps
            pred_minus = layer.forward(x)
            loss_minus = mse_loss(pred_minus, target).item()

            bias_grads[j] = (loss_plus - loss_minus) / (2 * eps * batch_size)
            layer.biases.data[j] = original

        return weight_grads, bias_grads

    def test_weight_gradients_linear(self) -> None:
        layer = DenseLayer(3, 2, activation="linear", weight_init="normal", seed=42)
        x = Tensor(np.random.randn(4, 3))
        target = Tensor(np.random.randn(4, 2))

        _ = layer.forward(x)
        pred = layer._output
        # Use MSE derivative (normalised by total elements)
        from robot.learning.losses import mse_derivative

        grad_out = mse_derivative(pred, target)  # type: ignore[arg-type]
        layer.backward(grad_out)

        numerical_w, numerical_b = self._numerical_gradient(layer, x, target)
        assert np.allclose(layer.weight_grad.data, numerical_w, atol=1e-4), (
            f"Weight grad mismatch:\n  analytical={layer.weight_grad.data}\n  numerical={numerical_w}"
        )
        assert np.allclose(layer.bias_grad.data, numerical_b, atol=1e-4), (
            f"Bias grad mismatch:\n  analytical={layer.bias_grad.data}\n  numerical={numerical_b}"
        )

    def test_weight_gradients_relu(self) -> None:
        layer = DenseLayer(3, 2, activation="relu", weight_init="normal", seed=42)
        # Use inputs that are mostly positive so ReLU is active
        x = Tensor(np.abs(np.random.randn(4, 3)) + 0.5)
        target = Tensor(np.random.randn(4, 2))

        _ = layer.forward(x)
        pred = layer._output
        from robot.learning.losses import mse_derivative

        grad_out = mse_derivative(pred, target)  # type: ignore[arg-type]
        layer.backward(grad_out)

        numerical_w, _ = self._numerical_gradient(layer, x, target)
        # Relax tolerance for ReLU because of the non-linearity
        assert np.allclose(layer.weight_grad.data, numerical_w, atol=1e-3)


class TestDenseLayerState:
    def test_save_load_round_trip(self) -> None:
        layer = DenseLayer(4, 3, activation="relu", seed=42)
        x = Tensor(np.random.randn(2, 4))
        original_output = layer.forward(x)

        state = layer.get_state()
        restored = DenseLayer.from_state(state)

        # Forward pass with restored layer should give same result
        restored_output = restored.forward(x)
        assert np.allclose(original_output.data, restored_output.data, atol=1e-10)

    def test_state_preserves_activation(self) -> None:
        layer = DenseLayer(4, 3, activation="sigmoid", seed=42)
        state = layer.get_state()
        assert state["activation"] == "sigmoid"
        restored = DenseLayer.from_state(state)
        assert restored.activation_name == "sigmoid"

    def test_state_preserves_dimensions(self) -> None:
        layer = DenseLayer(8, 5, activation="tanh", seed=42)
        state = layer.get_state()
        assert state["in_features"] == 8
        assert state["out_features"] == 5
        restored = DenseLayer.from_state(state)
        assert restored.in_features == 8
        assert restored.out_features == 5


class TestDenseLayerInit:
    def test_he_init(self) -> None:
        layer = DenseLayer(100, 50, activation="relu", weight_init="he", seed=42)
        # He init: std = sqrt(2/fan_in) ≈ 0.141
        std = np.std(layer.weights.data)
        expected_std = np.sqrt(2.0 / 100)
        assert abs(std - expected_std) < 0.05

    def test_xavier_init(self) -> None:
        layer = DenseLayer(100, 50, activation="sigmoid", weight_init="xavier", seed=42)
        # Xavier init: std = sqrt(2/(fan_in + fan_out)) ≈ 0.115
        std = np.std(layer.weights.data)
        expected_std = np.sqrt(2.0 / (100 + 50))
        assert abs(std - expected_std) < 0.05

    def test_reproducible_with_seed(self) -> None:
        l1 = DenseLayer(4, 3, seed=123)
        l2 = DenseLayer(4, 3, seed=123)
        assert np.allclose(l1.weights.data, l2.weights.data)
        assert np.allclose(l1.biases.data, l2.biases.data)

    def test_different_seeds_different_weights(self) -> None:
        l1 = DenseLayer(4, 3, seed=1)
        l2 = DenseLayer(4, 3, seed=2)
        assert not np.allclose(l1.weights.data, l2.weights.data)
