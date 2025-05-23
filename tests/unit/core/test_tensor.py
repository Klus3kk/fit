import numpy as np
import pytest

from core.tensor import Tensor


class TestTensor:
    def test_init(self):
        # Test with numpy array
        data = np.array([1.0, 2.0, 3.0])
        t = Tensor(data)
        assert np.array_equal(t.data, data)
        assert not t.requires_grad
        assert t.grad is None

        # Test with list
        t = Tensor([1.0, 2.0, 3.0])
        assert np.array_equal(t.data, data)

        # Test with requires_grad=True
        t = Tensor(data, requires_grad=True)
        assert t.requires_grad

    def test_add(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        c = a + b

        assert np.array_equal(c.data, np.array([5.0, 7.0, 9.0]))
        assert c.requires_grad

        # Test backward
        c.backward(np.array([1.0, 1.0, 1.0]))

        assert np.array_equal(a.grad, np.array([1.0, 1.0, 1.0]))
        assert np.array_equal(b.grad, np.array([1.0, 1.0, 1.0]))

        # Test add with scalar
        d = a + 5
        assert np.array_equal(d.data, np.array([6.0, 7.0, 8.0]))
        assert d.requires_grad

    def test_mul(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        c = a * b

        assert np.array_equal(c.data, np.array([4.0, 10.0, 18.0]))
        assert c.requires_grad

        # Test backward
        c.backward(np.array([1.0, 1.0, 1.0]))

        assert np.array_equal(a.grad, np.array([4.0, 5.0, 6.0]))
        assert np.array_equal(b.grad, np.array([1.0, 2.0, 3.0]))

        # Test mul with scalar
        d = a * 5
        assert np.array_equal(d.data, np.array([5.0, 10.0, 15.0]))
        assert d.requires_grad

    def test_matmul(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        c = a @ b

        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.array_equal(c.data, expected)
        assert c.requires_grad

        # Test backward
        c.backward(np.array([[1.0, 1.0], [1.0, 1.0]]))

        # Check gradients: dL/dA = dL/dC @ B.T, dL/dB = A.T @ dL/dC
        expected_grad_a = np.array([[11.0, 11.0], [11.0, 11.0]])
        expected_grad_b = np.array([[4.0, 4.0], [6.0, 6.0]])

        assert np.array_equal(a.grad, expected_grad_a)
        assert np.array_equal(b.grad, expected_grad_b)

    def test_sum(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a.sum()

        assert b.data == 10.0
        assert b.requires_grad

        # Test with axis
        c = a.sum(axis=0)
        assert np.array_equal(c.data, np.array([4.0, 6.0]))
        assert c.requires_grad

        # Test backward
        b.backward()
        # Gradient should be all ones
        assert np.array_equal(a.grad, np.ones_like(a.data))

        # Reset gradient for next test
        a.zero_grad()

        # Test backward with axis
        c.backward(np.array([1.0, 1.0]))
        # Gradient should still be all ones
        assert np.array_equal(a.grad, np.ones_like(a.data))

    def test_mean(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a.mean()

        assert b.data == 2.5
        assert b.requires_grad

        # Test backward
        b.backward()
        # Gradient should be 1/n for all elements
        expected_grad = np.ones_like(a.data) / a.data.size
        assert np.allclose(a.grad, expected_grad)

    def test_gradient_clipping(self):
        a = Tensor([10.0, 20.0, 30.0], requires_grad=True)
        a.grad = np.array([10.0, 10.0, 10.0])

        # Clip gradients
        a.clip_gradients(max_norm=5.0)

        # Calculate expected gradient
        orig_norm = np.sqrt(300.0)  # sqrt(10^2 + 10^2 + 10^2)
        expected = np.array([10.0, 10.0, 10.0]) * (5.0 / orig_norm)

        assert np.allclose(a.grad, expected)

    def test_relu(self):
        a = Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
        b = a.relu()

        # Test forward pass
        assert np.array_equal(b.data, np.array([0.0, 0.0, 1.0, 2.0]))

        # Test backward pass
        b.backward(np.array([1.0, 1.0, 1.0, 1.0]))
        expected_grad = np.array([0.0, 0.0, 1.0, 1.0])
        assert np.array_equal(a.grad, expected_grad)

    def test_exp(self):
        a = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        b = a.exp()

        # Test forward pass
        expected = np.array([1.0, np.exp(1), np.exp(2)])
        assert np.allclose(b.data, expected)

        # Test backward pass
        b.backward(np.array([1.0, 1.0, 1.0]))
        # Gradient is exp(x)
        assert np.allclose(a.grad, expected)

    def test_log(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = a.log()

        # Test forward pass
        expected = np.array([0.0, np.log(2), np.log(3)])
        assert np.allclose(b.data, expected)

        # Test backward pass
        b.backward(np.array([1.0, 1.0, 1.0]))
        # Gradient is 1/x
        expected_grad = np.array([1.0, 0.5, 1.0 / 3.0])
        assert np.allclose(a.grad, expected_grad)

    def test_reshape(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a.reshape((4,))

        # Test forward pass
        assert np.array_equal(b.data, np.array([1.0, 2.0, 3.0, 4.0]))

        # Test backward pass
        b.backward(np.array([1.0, 2.0, 3.0, 4.0]))
        expected_grad = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.array_equal(a.grad, expected_grad)

    def test_max(self):
        a = Tensor([[1.0, 3.0], [4.0, 2.0]], requires_grad=True)
        b = a.max()

        # Test forward pass
        assert b.data == 4.0

        # Test backward pass
        b.backward()
        # Only the max element should get gradient = 1.0
        expected_grad = np.array([[0.0, 0.0], [1.0, 0.0]])
        assert np.array_equal(a.grad, expected_grad)

        # Reset gradient
        a.zero_grad()

        # Test max with axis
        c = a.max(axis=1)
        assert np.array_equal(c.data, np.array([3.0, 4.0]))

        # Test backward with axis
        c.backward(np.array([1.0, 1.0]))
        # For each row, only the max element should get gradient
        expected_grad = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert np.array_equal(a.grad, expected_grad)
