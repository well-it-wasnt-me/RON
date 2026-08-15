"""Synthetic learning demonstration for the local neural network.

Trains small MLPs on simple problems and verifies that:
1. Loss decreases over training.
2. Trained predictions are better than untrained predictions.

This module is **not** connected to the robot.  It serves as a
standalone proof that the neural network framework works.
"""

from __future__ import annotations

import numpy as np

from robot.learning.network import MLP
from robot.learning.optimizers import Adam
from robot.learning.tensor import Tensor


def demo_xor() -> dict[str, object]:
    """Train an MLP on the XOR problem.

    Uses tanh output (range [-1, 1]) with MSE loss, which is more
    numerically stable than sigmoid for this problem.  Targets are
    mapped from {0, 1} to {-1, 1}.

    Returns a dict with training metrics.
    """
    np.random.default_rng(42)

    # XOR dataset with targets mapped to [-1, 1] for tanh
    x = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64))
    y = Tensor(np.array([[-1], [1], [1], [-1]], dtype=np.float64))

    model = MLP(
        input_size=2,
        hidden_sizes=[8, 8],
        output_size=1,
        activation="tanh",
        output_activation="tanh",
        weight_init="xavier",
        seed=42,
    )

    optimizer = Adam(learning_rate=0.01)

    # Record initial loss
    initial_loss, initial_pred = model.network.train_step(x, y, loss_fn="mse")

    losses: list[float] = [initial_loss]

    # Train
    for _epoch in range(500):
        loss, _ = model.network.train_step(x, y, loss_fn="mse", optimizer=optimizer)
        losses.append(loss)

    final_loss = losses[-1]
    final_pred = model.network.predict(x)

    # Map tanh outputs back to [0, 1] for comparison
    mapped_initial = (initial_pred.data + 1) / 2
    mapped_final = (final_pred.data + 1) / 2
    target_01 = (y.data + 1) / 2

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decreased": final_loss < initial_loss,
        "losses": losses,
        "initial_pred": mapped_initial.flatten().tolist(),
        "final_pred": mapped_final.flatten().tolist(),
        "target": target_01.flatten().tolist(),
        "param_count": model.network.param_count(),
    }


def demo_regression() -> dict[str, object]:
    """Train an MLP on a simple linear regression problem.

    ``y = 2*x + 1`` with noise.

    Returns a dict with training metrics.
    """
    rng = np.random.default_rng(123)
    n_samples = 64

    x_raw = rng.uniform(-1.0, 1.0, (n_samples, 1))
    y_raw = 2.0 * x_raw + 1.0 + rng.normal(0, 0.1, (n_samples, 1))

    x = Tensor(x_raw.astype(np.float64))
    y = Tensor(y_raw.astype(np.float64))

    model = MLP(
        input_size=1,
        hidden_sizes=[16, 16],
        output_size=1,
        activation="relu",
        output_activation="linear",
        weight_init="he",
        seed=123,
    )

    optimizer = Adam(learning_rate=0.01)

    initial_loss, _ = model.network.train_step(x, y, loss_fn="mse")

    losses: list[float] = [initial_loss]

    for _epoch in range(200):
        loss, _ = model.network.train_step(x, y, loss_fn="mse", optimizer=optimizer)
        losses.append(loss)

    final_loss = losses[-1]
    test_x = Tensor(np.array([[0.5]], dtype=np.float64))
    test_pred = model.network.predict(test_x)

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decreased": final_loss < initial_loss,
        "losses": losses,
        "test_input": 0.5,
        "test_prediction": test_pred.item(),
        "expected_approx": 2.0 * 0.5 + 1.0,
        "param_count": model.network.param_count(),
    }


if __name__ == "__main__":
    print("=== XOR Demo ===")
    xor_result = demo_xor()
    print(f"  Initial loss: {xor_result['initial_loss']:.6f}")
    print(f"  Final loss:   {xor_result['final_loss']:.6f}")
    print(f"  Loss decreased: {xor_result['loss_decreased']}")
    _final_pred = xor_result["final_pred"]
    assert isinstance(_final_pred, list)
    print(f"  Predictions: {[f'{float(p):.4f}' for p in _final_pred]}")
    print(f"  Targets:      {xor_result['target']}")
    print(f"  Parameters:   {xor_result['param_count']}")

    print()
    print("=== Regression Demo ===")
    reg_result = demo_regression()
    print(f"  Initial loss: {reg_result['initial_loss']:.6f}")
    print(f"  Final loss:   {reg_result['final_loss']:.6f}")
    print(f"  Loss decreased: {reg_result['loss_decreased']}")
    print(f"  Test input:   {reg_result['test_input']}")
    print(f"  Predicted:    {reg_result['test_prediction']:.4f}")
    print(f"  Expected:     ~{reg_result['expected_approx']:.4f}")
    print(f"  Parameters:   {reg_result['param_count']}")
