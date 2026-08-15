"""Tests for Network, MLP, and the training loop."""

import json
import tempfile
from pathlib import Path

import numpy as np

from robot.learning.layers import DenseLayer
from robot.learning.network import MLP, Network
from robot.learning.optimizers import SGD, Adam
from robot.learning.tensor import Tensor


class TestNetworkForward:
    def test_single_layer(self) -> None:
        layers = [DenseLayer(2, 1, activation="linear", weight_init="normal", seed=42)]
        net = Network(layers)
        x = Tensor([[1.0, 2.0]])
        y = net.forward(x)
        assert y.shape == (1, 1)

    def test_multi_layer(self) -> None:
        layers = [
            DenseLayer(4, 8, activation="relu", seed=1),
            DenseLayer(8, 4, activation="relu", seed=2),
            DenseLayer(4, 2, activation="linear", seed=3),
        ]
        net = Network(layers)
        x = Tensor(np.random.randn(3, 4))
        y = net.forward(x)
        assert y.shape == (3, 2)

    def test_param_count(self) -> None:
        layers = [
            DenseLayer(4, 8, activation="relu", seed=1),
            DenseLayer(8, 2, activation="linear", seed=2),
        ]
        net = Network(layers)
        # 4*8 + 8 + 8*2 + 2 = 32+8+16+2 = 58
        assert net.param_count() == 58


class TestNetworkBackward:
    def test_backward_shapes(self) -> None:
        layers = [
            DenseLayer(4, 8, activation="relu", seed=1),
            DenseLayer(8, 2, activation="linear", seed=2),
        ]
        net = Network(layers)
        x = Tensor(np.random.randn(5, 4))
        net.forward(x)
        grad = Tensor(np.random.randn(5, 2))
        grad_in = net.backward(grad)
        assert grad_in.shape == (5, 4)


class TestMLP:
    def test_default_construction(self) -> None:
        mlp = MLP(input_size=4, hidden_sizes=[8, 8], output_size=2, seed=42)
        x = Tensor(np.random.randn(3, 4))
        y = mlp.forward(x)
        assert y.shape == (3, 2)

    def test_param_count(self) -> None:
        mlp = MLP(input_size=4, hidden_sizes=[8], output_size=2, seed=42)
        # 4*8+8 + 8*2+2 = 42+18 = 58
        # Wait: 4*8=32 weights + 8 biases + 8*2=16 weights + 2 biases = 58
        assert mlp.network.param_count() == 58

    def test_tanh_output_range(self) -> None:
        mlp = MLP(
            input_size=2,
            hidden_sizes=[4],
            output_size=1,
            activation="relu",
            output_activation="tanh",
            seed=42,
        )
        x = Tensor(np.random.randn(10, 2))
        y = mlp.forward(x)
        # Tanh output should be in [-1, 1]
        assert np.all(y.data >= -1) and np.all(y.data <= 1)

    def test_sigmoid_output_range(self) -> None:
        mlp = MLP(
            input_size=2,
            hidden_sizes=[4],
            output_size=1,
            activation="relu",
            output_activation="sigmoid",
            seed=42,
        )
        x = Tensor(np.random.randn(10, 2))
        y = mlp.forward(x)
        # Sigmoid output should be in (0, 1)
        assert np.all(y.data > 0) and np.all(y.data < 1)


class TestTrainingStep:
    def test_mse_loss_decreases(self) -> None:
        """Verify that training reduces MSE loss on a simple problem."""
        rng = np.random.default_rng(42)
        x = Tensor(rng.uniform(-1, 1, (32, 2)).astype(np.float64))
        y = Tensor((x.data[:, 0:1] * 2.0 + 1.0).astype(np.float64))

        mlp = MLP(input_size=2, hidden_sizes=[8, 8], output_size=1, seed=42)
        optimizer = Adam(learning_rate=0.01)

        initial_loss, _ = mlp.network.train_step(x, y, loss_fn="mse", optimizer=optimizer)

        for _ in range(100):
            mlp.network.train_step(x, y, loss_fn="mse", optimizer=optimizer)

        final_loss, _ = mlp.network.train_step(x, y, loss_fn="mse")
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.6f}, final={final_loss:.6f}"
        )

    def test_xor_learns(self) -> None:
        """Verify that XOR can be learned with tanh output and MSE loss.

        Targets are mapped to [-1, 1] for tanh output.
        """
        x = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64))
        y = Tensor(np.array([[-1], [1], [1], [-1]], dtype=np.float64))

        mlp = MLP(
            input_size=2,
            hidden_sizes=[16, 16],
            output_size=1,
            activation="tanh",
            output_activation="tanh",
            weight_init="xavier",
            seed=42,
        )
        optimizer = Adam(learning_rate=0.01)

        for _ in range(1000):
            mlp.network.train_step(x, y, loss_fn="mse", optimizer=optimizer)

        pred = mlp.network.predict(x)
        # Map tanh output to [0, 1]: (pred + 1) / 2
        pred_01 = (pred.data + 1) / 2
        # XOR(0,0)≈0, XOR(0,1)≈1, XOR(1,0)≈1, XOR(1,1)≈0
        assert pred_01[0, 0] < 0.3, f"XOR(0,0) = {pred_01[0, 0]:.4f}, expected <0.3"
        assert pred_01[1, 0] > 0.7, f"XOR(0,1) = {pred_01[1, 0]:.4f}, expected >0.7"
        assert pred_01[2, 0] > 0.7, f"XOR(1,0) = {pred_01[2, 0]:.4f}, expected >0.7"
        assert pred_01[3, 0] < 0.3, f"XOR(1,1) = {pred_01[3, 0]:.4f}, expected <0.3"


class TestSGDOptimizer:
    def test_sgd_reduces_loss(self) -> None:
        rng = np.random.default_rng(42)
        x = Tensor(rng.uniform(-1, 1, (32, 2)).astype(np.float64))
        y = Tensor((x.data[:, 0:1] * 3.0 + 0.5).astype(np.float64))

        mlp = MLP(input_size=2, hidden_sizes=[16], output_size=1, seed=42)
        optimizer = SGD(learning_rate=0.01)

        initial_loss, _ = mlp.network.train_step(x, y, loss_fn="mse", optimizer=optimizer)
        for _ in range(200):
            mlp.network.train_step(x, y, loss_fn="mse", optimizer=optimizer)
        final_loss, _ = mlp.network.train_step(x, y, loss_fn="mse")

        assert final_loss < initial_loss

    def test_sgd_with_momentum(self) -> None:
        rng = np.random.default_rng(42)
        x = Tensor(rng.uniform(-1, 1, (32, 2)).astype(np.float64))
        y = Tensor((x.data[:, 0:1] * 3.0 + 0.5).astype(np.float64))

        mlp = MLP(input_size=2, hidden_sizes=[16], output_size=1, seed=42)
        optimizer = SGD(learning_rate=0.01, momentum=0.9)

        initial_loss, _ = mlp.network.train_step(x, y, loss_fn="mse", optimizer=optimizer)
        for _ in range(200):
            mlp.network.train_step(x, y, loss_fn="mse", optimizer=optimizer)
        final_loss, _ = mlp.network.train_step(x, y, loss_fn="mse")

        assert final_loss < initial_loss


class TestAdamOptimizer:
    def test_adam_reduces_loss(self) -> None:
        rng = np.random.default_rng(42)
        x = Tensor(rng.uniform(-1, 1, (64, 3)).astype(np.float64))
        y = Tensor((x.data[:, 0:1] + x.data[:, 1:2] - x.data[:, 2:3]).astype(np.float64))

        mlp = MLP(input_size=3, hidden_sizes=[16, 16], output_size=1, seed=42)
        optimizer = Adam(learning_rate=0.001)

        initial_loss, _ = mlp.network.train_step(x, y, loss_fn="mse", optimizer=optimizer)
        for _ in range(500):
            mlp.network.train_step(x, y, loss_fn="mse", optimizer=optimizer)
        final_loss, _ = mlp.network.train_step(x, y, loss_fn="mse")

        assert final_loss < initial_loss


class TestModelSaveLoad:
    def test_save_load_round_trip(self) -> None:
        mlp = MLP(input_size=4, hidden_sizes=[8, 8], output_size=2, seed=42)
        x = Tensor(np.random.randn(3, 4))
        original_pred = mlp.network.predict(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            mlp.save(path)

            assert path.exists()

            loaded = MLP.load(path)
            loaded_pred = loaded.network.predict(x)

            assert np.allclose(original_pred.data, loaded_pred.data, atol=1e-10)

    def test_save_creates_parent_dirs(self) -> None:
        mlp = MLP(input_size=2, hidden_sizes=[4], output_size=1, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "model.json"
            mlp.save(path)
            assert path.exists()

    def test_save_load_preserves_dimensions(self) -> None:
        mlp = MLP(input_size=6, hidden_sizes=[10, 8], output_size=3, seed=42)
        x = Tensor(np.random.randn(2, 6))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            mlp.save(path)
            loaded = MLP.load(path)
            y = loaded.network.predict(x)
            assert y.shape == (2, 3)

    def test_load_file_content(self) -> None:
        mlp = MLP(input_size=2, hidden_sizes=[4], output_size=1, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            mlp.save(path)
            state = json.loads(path.read_text())
            assert "layers" in state
            assert len(state["layers"]) == 2  # 1 hidden + 1 output


class TestNetworkRepr:
    def test_repr(self) -> None:
        mlp = MLP(input_size=2, hidden_sizes=[4, 4], output_size=1, seed=42)
        r = repr(mlp.network)
        assert "Network" in r
        assert "params=" in r


class TestCrossEntropyTraining:
    def test_classification_learns(self) -> None:
        """Simple 2-class classification with cross-entropy loss."""
        rng = np.random.default_rng(42)
        # Two clusters: class 0 around (-1, -1), class 1 around (1, 1)
        n = 50
        x0 = rng.normal(-1, 0.5, (n, 2))
        x1 = rng.normal(1, 0.5, (n, 2))
        x = Tensor(np.vstack([x0, x1]).astype(np.float64))
        # One-hot targets
        y = Tensor(np.zeros((2 * n, 2), dtype=np.float64))
        y.data[:n, 0] = 1.0
        y.data[n:, 1] = 1.0

        mlp = MLP(
            input_size=2,
            hidden_sizes=[16, 16],
            output_size=2,
            activation="relu",
            output_activation="softmax",
            seed=42,
        )
        optimizer = Adam(learning_rate=0.005)

        initial_loss, _ = mlp.network.train_step(x, y, loss_fn="cross_entropy", optimizer=optimizer)
        for _ in range(200):
            mlp.network.train_step(x, y, loss_fn="cross_entropy", optimizer=optimizer)
        final_loss, _ = mlp.network.train_step(x, y, loss_fn="cross_entropy")

        assert final_loss < initial_loss, (
            f"Cross-entropy loss did not decrease: initial={initial_loss:.6f}, final={final_loss:.6f}"
        )
