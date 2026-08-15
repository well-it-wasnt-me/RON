"""Tests for activation functions."""

import numpy as np
import pytest

from robot.learning.activations import (
    get_activation,
    linear,
    linear_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    softmax,
    tanh,
    tanh_derivative,
)
from robot.learning.tensor import Tensor


class TestReLU:
    def test_positive(self) -> None:
        x = Tensor([1.0, 2.0, 3.0])
        y = relu(x)
        assert np.allclose(y.data, [1.0, 2.0, 3.0])

    def test_negative(self) -> None:
        x = Tensor([-1.0, -2.0, -3.0])
        y = relu(x)
        assert np.allclose(y.data, [0.0, 0.0, 0.0])

    def test_mixed(self) -> None:
        x = Tensor([-2.0, 0.0, 3.0])
        y = relu(x)
        assert np.allclose(y.data, [0.0, 0.0, 3.0])

    def test_derivative_positive(self) -> None:
        x = Tensor([1.0, 2.0, 3.0])
        d = relu_derivative(x)
        assert np.allclose(d.data, [1.0, 1.0, 1.0])

    def test_derivative_negative(self) -> None:
        x = Tensor([-1.0, -2.0, -3.0])
        d = relu_derivative(x)
        assert np.allclose(d.data, [0.0, 0.0, 0.0])


class TestSigmoid:
    def test_zero(self) -> None:
        x = Tensor([0.0])
        y = sigmoid(x)
        assert np.isclose(y.data[0], 0.5)

    def test_large_positive(self) -> None:
        x = Tensor([100.0])
        y = sigmoid(x)
        assert np.isclose(y.data[0], 1.0, atol=1e-4)

    def test_large_negative(self) -> None:
        x = Tensor([-100.0])
        y = sigmoid(x)
        assert np.isclose(y.data[0], 0.0, atol=1e-4)

    def test_derivative_at_half(self) -> None:
        s = sigmoid(Tensor([0.0]))
        d = sigmoid_derivative(s)
        # sigma'(0) = 0.5 * 0.5 = 0.25
        assert np.isclose(d.data[0], 0.25)

    def test_derivative_range(self) -> None:
        s = sigmoid(Tensor([0.0, 5.0, -5.0]))
        d = sigmoid_derivative(s)
        # All derivatives should be in (0, 0.25]
        assert np.all(d.data > 0)
        assert np.all(d.data <= 0.25)


class TestTanh:
    def test_zero(self) -> None:
        x = Tensor([0.0])
        y = tanh(x)
        assert np.isclose(y.data[0], 0.0, atol=1e-6)

    def test_saturates(self) -> None:
        x = Tensor([10.0, -10.0])
        y = tanh(x)
        assert np.isclose(y.data[0], 1.0, atol=1e-4)
        assert np.isclose(y.data[1], -1.0, atol=1e-4)

    def test_derivative_at_zero(self) -> None:
        t = tanh(Tensor([0.0]))
        d = tanh_derivative(t)
        # tanh'(0) = 1 - 0^2 = 1
        assert np.isclose(d.data[0], 1.0)


class TestLinear:
    def test_identity(self) -> None:
        x = Tensor([1.0, -2.0, 3.0])
        y = linear(x)
        assert np.allclose(y.data, x.data)

    def test_derivative(self) -> None:
        x = Tensor([1.0, 2.0])
        d = linear_derivative(x)
        assert np.allclose(d.data, [1.0, 1.0])


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        x = Tensor([[1.0, 2.0, 3.0]])
        y = softmax(x)
        assert np.isclose(y.data.sum(), 1.0, atol=1e-6)

    def test_largest_input_highest_prob(self) -> None:
        x = Tensor([[1.0, 5.0, 2.0]])
        y = softmax(x)
        assert y.data[0, 1] > y.data[0, 0]
        assert y.data[0, 1] > y.data[0, 2]

    def test_batch(self) -> None:
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = softmax(x)
        assert y.shape == (2, 2)
        for i in range(2):
            assert np.isclose(y.data[i].sum(), 1.0, atol=1e-6)


class TestGetActivation:
    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown activation"):
            get_activation("unknown")

    def test_all_known(self) -> None:
        for name in ("relu", "sigmoid", "tanh", "linear", "softmax"):
            fn, deriv = get_activation(name)
            assert callable(fn)
            assert callable(deriv)
