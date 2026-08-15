"""Tests for loss functions."""

import numpy as np
import pytest

from robot.learning.losses import (
    cross_entropy_derivative,
    cross_entropy_loss,
    get_loss,
    mse_derivative,
    mse_loss,
)
from robot.learning.tensor import Tensor


class TestMSE:
    def test_perfect_prediction(self) -> None:
        pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = Tensor([[1.0, 2.0], [3.0, 4.0]])
        loss = mse_loss(pred, target)
        assert np.isclose(loss.item(), 0.0)

    def test_nonzero_loss(self) -> None:
        pred = Tensor([[1.0, 2.0]])
        target = Tensor([[3.0, 4.0]])
        loss = mse_loss(pred, target)
        # MSE = mean((2^2 + 2^2)) = mean(8) = 4.0
        assert np.isclose(loss.item(), 4.0)

    def test_derivative_direction(self) -> None:
        pred = Tensor([[1.0, 2.0]])
        target = Tensor([[3.0, 4.0]])
        grad = mse_derivative(pred, target)
        # Gradient should point from pred toward target
        assert np.all(grad.data < 0)  # pred < target, so grad is negative (pred - target < 0)

    def test_derivative_magnitude(self) -> None:
        pred = Tensor([[1.0, 2.0]])
        target = Tensor([[3.0, 4.0]])
        grad = mse_derivative(pred, target)
        # dL/dpred = 2*(pred - target) / n
        n = pred.size
        expected = 2.0 * (pred.data - target.data) / n
        assert np.allclose(grad.data, expected)

    def test_batch(self) -> None:
        pred = Tensor([[1.0], [2.0]])
        target = Tensor([[1.0], [4.0]])
        loss = mse_loss(pred, target)
        # (0 + 4) / 2 = 2.0
        assert np.isclose(loss.item(), 2.0)


class TestCrossEntropy:
    def test_perfect_prediction(self) -> None:
        pred = Tensor([[0.9, 0.1]])
        target = Tensor([[1.0, 0.0]])
        loss = cross_entropy_loss(pred, target)
        # Should be small (close to -log(0.9))
        assert loss.item() < 0.2

    def test_bad_prediction(self) -> None:
        pred = Tensor([[0.1, 0.9]])
        target = Tensor([[1.0, 0.0]])
        loss = cross_entropy_loss(pred, target)
        # Should be large (-log(0.1) ≈ 2.3)
        assert loss.item() > 1.0

    def test_derivative_direction(self) -> None:
        pred = Tensor([[0.8, 0.2]])
        target = Tensor([[1.0, 0.0]])
        grad = cross_entropy_derivative(pred, target)
        # pred - target = [-0.2, 0.2] (divided by batch size)
        # Should be negative for over-predicted, positive for under-predicted
        assert grad.data[0, 0] < 0  # over-predicted class
        assert grad.data[0, 1] > 0  # under-predicted class


class TestGetLoss:
    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown loss"):
            get_loss("unknown")

    def test_all_known(self) -> None:
        for name in ("mse", "cross_entropy"):
            fn, deriv = get_loss(name)
            assert callable(fn)
            assert callable(deriv)
