"""
Tests for the autograd engine.
"""

import numpy as np
import pytest

from core.autograd import (
    Node,
    Function,
    Add,
    Multiply,
    MatMul,
    Sum,
    Exp,
    Log,
    Reshape,
    ReLU,
    Mean,
)
import core.autograd as autograd


class TestNode:
    """Tests for the Node class."""

    def test_init(self):
        """Test Node initialization."""
        node = Node()
        assert node.parents == set()
        assert node.grad is None
        assert not node.requires_grad

        node_with_grad = Node(requires_grad=True)
        assert node_with_grad.requires_grad


class TestFunctions:
    """Tests for the autograd functions."""

    def test_add(self):
        """Test the Add function."""
        # Test forward pass
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        ctx = {}
        result = Add.apply(ctx, a, b)
        assert np.array_equal(result, np.array([5.0, 7.0, 9.0]))

        # Test backward pass
        grad_output = np.array([1.0, 1.0, 1.0])
        grad_a, grad_b = Add.backward(ctx, grad_output)
        assert np.array_equal(grad_a, grad_output)
        assert np.array_equal(grad_b, grad_output)

    def test_multiply(self):
        """Test the Multiply function."""
        # Test forward pass
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        ctx = {}
        result = Multiply.apply(ctx, a, b)
        assert np.array_equal(result, np.array([4.0, 10.0, 18.0]))

        # Test backward pass
        grad_output = np.array([1.0, 1.0, 1.0])
        grad_a, grad_b = Multiply.backward(ctx, grad_output)
        assert np.array_equal(grad_a, b)
        assert np.array_equal(grad_b, a)

    def test_matmul(self):
        """Test the MatMul function."""
        # Test forward pass
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        ctx = {}
        result = MatMul.apply(ctx, a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.array_equal(result, expected)

        # Test backward pass
        grad_output = np.array([[1.0, 1.0], [1.0, 1.0]])
        grad_a, grad_b = MatMul.backward(ctx, grad_output)
        expected_grad_a = np.array([[11.0, 11.0], [11.0, 11.0]])
        expected_grad_b = np.array([[4.0, 4.0], [6.0, 6.0]])
        assert np.array_equal(grad_a, expected_grad_a)
        assert np.array_equal(grad_b, expected_grad_b)

    def test_sum(self):
        """Test the Sum function."""
        # Test forward pass
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        ctx = {}
        result = Sum.apply(ctx, a)
        assert result == 10.0

        # Test forward pass with axis
        ctx = {}
        result = Sum.apply(ctx, a, axis=0)
        assert np.array_equal(result, np.array([4.0, 6.0]))

        # Test backward pass
        grad_output = np.array(1.0)
        grad_a, _, _ = Sum.backward(ctx, grad_output)
        assert np.array_equal(grad_a, np.ones_like(a))

        # Test backward pass with axis
        ctx = {"input_shape": a.shape, "axis": 0, "keepdims": False}
        grad_output = np.array([1.0, 1.0])
        grad_a, _, _ = Sum.backward(ctx, grad_output)
        assert np.array_equal(grad_a, np.ones_like(a))

    def test_exp(self):
        """Test the Exp function."""
        # Test forward pass
        a = np.array([0.0, 1.0, 2.0])
        ctx = {}
        result = Exp.apply(ctx, a)
        expected = np.array([1.0, np.exp(1), np.exp(2)])
        assert np.allclose(result, expected)

        # Test backward pass
        grad_output = np.array([1.0, 1.0, 1.0])
        (grad_a,) = Exp.backward(ctx, grad_output)
        assert np.allclose(grad_a, expected)

    def test_log(self):
        """Test the Log function."""
        # Test forward pass
        a = np.array([1.0, 2.0, 3.0])
        ctx = {}
        result = Log.apply(ctx, a)
        expected = np.array([0.0, np.log(2), np.log(3)])
        assert np.allclose(result, expected)

        # Test backward pass
        grad_output = np.array([1.0, 1.0, 1.0])
        (grad_a,) = Log.backward(ctx, grad_output)
        expected_grad = np.array([1.0, 0.5, 1 / 3])
        assert np.allclose(grad_a, expected_grad)

    def test_reshape(self):
        """Test the Reshape function."""
        # Test forward pass
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        new_shape = (4,)
        ctx = {}
        result = Reshape.apply(ctx, a, new_shape)
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.array_equal(result, expected)

        # Test backward pass
        grad_output = np.array([1.0, 1.0, 1.0, 1.0])
        grad_a, _ = Reshape.backward(ctx, grad_output)
        expected_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert np.array_equal(grad_a, expected_grad)

    def test_relu(self):
        """Test the ReLU function."""
        # Test forward pass
        a = np.array([-1.0, 0.0, 1.0, 2.0])
        ctx = {}
        result = ReLU.apply(ctx, a)
        expected = np.array([0.0, 0.0, 1.0, 2.0])
        assert np.array_equal(result, expected)

        # Test backward pass
        grad_output = np.array([1.0, 1.0, 1.0, 1.0])
        (grad_a,) = ReLU.backward(ctx, grad_output)
        expected_grad = np.array([0.0, 0.0, 1.0, 1.0])
        assert np.array_equal(grad_a, expected_grad)

    def test_mean(self):
        """Test the Mean function."""
        # Test forward pass
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        ctx = {}
        result = Mean.apply(ctx, a)
        assert result == 2.5

        # Test forward pass with axis
        ctx = {}
        result = Mean.apply(ctx, a, axis=0)
        assert np.array_equal(result, np.array([2.0, 3.0]))

        # Test backward pass
        grad_output = np.array(1.0)
        grad_a, _, _ = Mean.backward(ctx, grad_output)
        # Gradient should be 1/n where n is the number of elements
        expected_grad = np.ones_like(a) / a.size
        assert np.array_equal(grad_a, expected_grad)

        # Test backward pass with axis
        ctx = {"input_shape": a.shape, "axis": 0, "keepdims": False, "size": 2}
        grad_output = np.array([1.0, 1.0])
        grad_a, _, _ = Mean.backward(ctx, grad_output)
        # Gradient should be 1/m where m is the size of the reduced dimension
        expected_grad = np.ones_like(a) / 2
        assert np.array_equal(grad_a, expected_grad)


class TestAutogradRegistry:
    """Tests for the autograd function registry."""

    def test_get_function(self):
        """Test getting functions from the registry."""
        assert autograd.get_function("add") == Add
        assert autograd.get_function("multiply") == Multiply
        assert autograd.get_function("matmul") == MatMul

        with pytest.raises(ValueError):
            autograd.get_function("nonexistent_function")
