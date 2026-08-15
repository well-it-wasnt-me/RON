"""Tests for the Tensor wrapper."""

import numpy as np

from robot.learning.tensor import Tensor


class TestTensorCreation:
    def test_from_list(self) -> None:
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.size == 3

    def test_from_numpy(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        t = Tensor(arr)
        assert t.shape == (2, 2)
        assert t.ndim == 2

    def test_from_tensor(self) -> None:
        t1 = Tensor([1.0, 2.0])
        t2 = Tensor(t1)
        assert t2.shape == t1.shape
        # Ensure independence
        id(t1._data)
        id(t2._data)

    def test_zeros(self) -> None:
        t = Tensor.zeros(3, 4)
        assert t.shape == (3, 4)
        assert np.allclose(t.data, 0.0)

    def test_ones(self) -> None:
        t = Tensor.ones(2, 3)
        assert t.shape == (2, 3)
        assert np.allclose(t.data, 1.0)

    def test_randn(self) -> None:
        t = Tensor.randn(100, seed=42)
        assert t.shape == (100,)
        assert not np.allclose(t.data, 0.0)

    def test_randn_reproducible(self) -> None:
        t1 = Tensor.randn(10, seed=99)
        t2 = Tensor.randn(10, seed=99)
        assert np.allclose(t1.data, t2.data)

    def test_uniform(self) -> None:
        t = Tensor.uniform(-1.0, 1.0, 100, seed=42)
        assert t.shape == (100,)
        assert np.all(t.data >= -1.0)
        assert np.all(t.data <= 1.0)

    def test_from_row(self) -> None:
        t = Tensor.from_row([5.0, 6.0, 7.0])
        assert t.shape == (3,)


class TestTensorArithmetic:
    def test_add_tensor(self) -> None:
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = a + b
        assert np.allclose(c.data, [4.0, 6.0])

    def test_add_scalar(self) -> None:
        a = Tensor([1.0, 2.0])
        c = a + 5.0
        assert np.allclose(c.data, [6.0, 7.0])

    def test_radd_scalar(self) -> None:
        a = Tensor([1.0, 2.0])
        c = 5.0 + a
        assert np.allclose(c.data, [6.0, 7.0])

    def test_sub_tensor(self) -> None:
        a = Tensor([5.0, 3.0])
        b = Tensor([1.0, 2.0])
        c = a - b
        assert np.allclose(c.data, [4.0, 1.0])

    def test_mul_tensor(self) -> None:
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = a * b
        assert np.allclose(c.data, [8.0, 15.0])

    def test_mul_scalar(self) -> None:
        a = Tensor([2.0, 3.0])
        c = a * 2.0
        assert np.allclose(c.data, [4.0, 6.0])

    def test_rmul_scalar(self) -> None:
        a = Tensor([2.0, 3.0])
        c = 2.0 * a
        assert np.allclose(c.data, [4.0, 6.0])

    def test_div_scalar(self) -> None:
        a = Tensor([4.0, 6.0])
        c = a / 2.0
        assert np.allclose(c.data, [2.0, 3.0])

    def test_neg(self) -> None:
        a = Tensor([1.0, -2.0])
        c = -a
        assert np.allclose(c.data, [-1.0, 2.0])

    def test_matmul(self) -> None:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = a @ b
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(c.data, expected)


class TestTensorOperations:
    def test_sum(self) -> None:
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        s = t.sum()
        assert np.isclose(s.item(), 10.0)

    def test_mean(self) -> None:
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        m = t.mean()
        assert np.isclose(m.item(), 2.5)

    def test_reshape(self) -> None:
        t = Tensor([1.0, 2.0, 3.0, 4.0])
        r = t.reshape(2, 2)
        assert r.shape == (2, 2)

    def test_transpose(self) -> None:
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        tt = t.T()
        assert tt.shape == (2, 2)
        assert np.allclose(tt.data, t.data.T)

    def test_clip(self) -> None:
        t = Tensor([-2.0, 0.5, 3.0])
        c = t.clip(-1.0, 1.0)
        assert np.allclose(c.data, [-1.0, 0.5, 1.0])

    def test_flatten(self) -> None:
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        f = t.flatten()
        assert f.shape == (4,)

    def test_item(self) -> None:
        t = Tensor([5.0])
        assert t.item() == 5.0

    def test_len(self) -> None:
        t = Tensor([1.0, 2.0, 3.0])
        assert len(t) == 3

    def test_getitem(self) -> None:
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        row = t[0]
        assert np.allclose(row.data, [1.0, 2.0])

    def test_setitem(self) -> None:
        t = Tensor.zeros(3)
        t[1] = 5.0
        assert np.isclose(t.data[1], 5.0)

    def test_repr(self) -> None:
        t = Tensor([1.0])
        assert "Tensor" in repr(t)

    def test_eq(self) -> None:
        a = Tensor([1.0, 2.0])
        b = Tensor([1.0, 2.0])
        assert a == b
