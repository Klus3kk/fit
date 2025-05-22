"""
Tests for the ops module.
"""

import numpy as np
import pytest

from core.ops import (
    matmul,
    transpose,
    sigmoid,
    softmax,
    tanh,
    zeros,
    ones,
    randn,
    concatenate,
    abs,
    sqrt,
    sin,
    cos,
    std,
    var,
    argmax,
    logsumexp,
    mse_loss,
    binary_cross_entropy,
    softmax_cross_entropy,
    cosine_similarity,
    pairwise_distance,
)
from core.tensor import Tensor


class TestMatrixOps:
    """Test matrix operations."""

    def test_matmul(self):
        """Test matrix multiplication."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        c = matmul(a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.array_equal(c.data, expected)

        # Test backward
        c.backward(np.array([[1.0, 1.0], [1.0, 1.0]]))

        # dC/dA = dC/dC @ B.T, dC/dB = A.T @ dC/dC
        expected_grad_a = np.array([[11.0, 11.0], [11.0, 11.0]])
        expected_grad_b = np.array([[4.0, 4.0], [6.0, 6.0]])

        assert np.array_equal(a.grad, expected_grad_a)
        assert np.array_equal(b.grad, expected_grad_b)

    def test_transpose(self):
        """Test tensor transpose."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

        # Test default transpose (reverse dims)
        b = transpose(a)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        assert np.array_equal(b.data, expected)

        # Test backward
        b.backward(np.ones_like(b.data))
        assert np.array_equal(a.grad, np.ones_like(a.data))

        # Reset gradients
        a.zero_grad()

        # Test transpose with specific axes
        c = transpose(a, (1, 0))
        assert np.array_equal(c.data, expected)

        # Test backward with specific axes
        c.backward(np.ones_like(c.data))
        assert np.array_equal(a.grad, np.ones_like(a.data))


class TestActivations:
    """Test activation functions."""

    def test_sigmoid(self):
        """Test sigmoid activation."""
        a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)

        b = sigmoid(a)
        expected = 1.0 / (1.0 + np.exp(-a.data))
        assert np.allclose(b.data, expected)

        # Test backward
        b.backward(np.array([1.0, 1.0, 1.0]))

        # Gradient: sigmoid(x) * (1 - sigmoid(x))
        expected_grad = expected * (1 - expected)
        assert np.allclose(a.grad, expected_grad)

    def test_softmax(self):
        """Test softmax activation."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

        b = softmax(a, axis=1)

        # Expected softmax values
        exp_a = np.exp(a.data - np.max(a.data, axis=1, keepdims=True))
        expected = exp_a / np.sum(exp_a, axis=1, keepdims=True)

        assert np.allclose(b.data, expected)
        assert np.allclose(np.sum(b.data, axis=1), np.array([1.0, 1.0]))

        # Test backward
        b.backward(np.ones_like(b.data))

        # For a simple check, gradient should sum to 0 along softmax axis
        assert np.allclose(np.sum(a.grad, axis=1), np.zeros(a.data.shape[0]))

    def test_tanh(self):
        """Test hyperbolic tangent activation."""
        a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)

        b = tanh(a)
        expected = np.tanh(a.data)
        assert np.allclose(b.data, expected)

        # Test backward
        b.backward(np.array([1.0, 1.0, 1.0]))

        # Gradient: 1 - tanh^2(x)
        expected_grad = 1 - expected**2
        assert np.allclose(a.grad, expected_grad)


class TestCreationOps:
    """Test tensor creation operations."""

    def test_zeros(self):
        """Test zeros creation."""
        a = zeros((2, 3), requires_grad=True)

        assert a.requires_grad
        assert a.data.shape == (2, 3)
        assert np.all(a.data == 0)

    def test_ones(self):
        """Test ones creation."""
        a = ones((2, 3), requires_grad=True)

        assert a.requires_grad
        assert a.data.shape == (2, 3)
        assert np.all(a.data == 1)

    def test_randn(self):
        """Test random normal creation."""
        a = randn((2, 3), requires_grad=True)

        assert a.requires_grad
        assert a.data.shape == (2, 3)

        # Check statistical properties (mean close to 0, std close to 1)
        assert -0.5 < np.mean(a.data) < 0.5
        assert 0.5 < np.std(a.data) < 1.5

    def test_concatenate(self):
        """Test tensor concatenation."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        # Test concatenation along axis 0
        c = concatenate([a, b], axis=0)
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        assert np.array_equal(c.data, expected)

        # Test backward
        c.backward(np.ones_like(c.data))
        assert np.array_equal(a.grad, np.ones_like(a.data))
        assert np.array_equal(b.grad, np.ones_like(b.data))

        # Reset gradients
        a.zero_grad()
        b.zero_grad()

        # Test concatenation along axis 1
        d = concatenate([a, b], axis=1)
        expected = np.array([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]])
        assert np.array_equal(d.data, expected)

        # Test backward
        d.backward(np.ones_like(d.data))
        assert np.array_equal(a.grad, np.ones_like(a.data))
        assert np.array_equal(b.grad, np.ones_like(b.data))


class TestMathOps:
    """Test mathematical operations."""

    def test_abs(self):
        """Test absolute value."""
        a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)

        b = abs(a)
        expected = np.array([1.0, 0.0, 1.0])
        assert np.array_equal(b.data, expected)

        # Test backward
        b.backward(np.array([1.0, 1.0, 1.0]))

        # Gradient: sign(x)
        expected_grad = np.array([-1.0, 0.0, 1.0])
        assert np.array_equal(a.grad, expected_grad)

    def test_sqrt(self):
        """Test square root."""
        a = Tensor([1.0, 4.0, 9.0], requires_grad=True)

        b = sqrt(a)
        expected = np.array([1.0, 2.0, 3.0])
        assert np.array_equal(b.data, expected)

        # Test backward
        b.backward(np.array([1.0, 1.0, 1.0]))

        # Gradient: 1 / (2 * sqrt(x))
        expected_grad = 1.0 / (2.0 * expected)
        assert np.allclose(a.grad, expected_grad)

    def test_sin(self):
        """Test sine function."""
        a = Tensor([0.0, np.pi / 2, np.pi], requires_grad=True)

        b = sin(a)
        expected = np.array([0.0, 1.0, 0.0])
        assert np.allclose(b.data, expected, rtol=1e-6)

        # Test backward
        b.backward(np.array([1.0, 1.0, 1.0]))

        # Gradient: cos(x)
        expected_grad = np.array([1.0, 0.0, -1.0])
        assert np.allclose(a.grad, expected_grad, rtol=1e-6)

    def test_cos(self):
        """Test cosine function."""
        a = Tensor([0.0, np.pi / 2, np.pi], requires_grad=True)

        b = cos(a)
        expected = np.array([1.0, 0.0, -1.0])
        assert np.allclose(b.data, expected, rtol=1e-6)

        # Test backward
        b.backward(np.array([1.0, 1.0, 1.0]))

        # Gradient: -sin(x)
        expected_grad = np.array([0.0, -1.0, 0.0])
        assert np.allclose(a.grad, expected_grad, rtol=1e-6)


class TestStatOps:
    """Test statistical operations."""

    def test_std(self):
        """Test standard deviation."""
        a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)

        b = std(a)
        expected = np.std(a.data)
        assert np.isclose(b.data, expected)

        # Test std with axis
        c = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        d = std(c, axis=0)
        expected = np.std(c.data, axis=0)
        assert np.allclose(d.data, expected)

    def test_var(self):
        """Test variance."""
        a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)

        b = var(a)
        expected = np.var(a.data)
        assert np.isclose(b.data, expected)

        # Test var with axis
        c = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        d = var(c, axis=0)
        expected = np.var(c.data, axis=0)
        assert np.allclose(d.data, expected)

    def test_logsumexp(self):
        """Test log-sum-exp."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)

        b = logsumexp(a)

        # Manual calculation of logsumexp
        a_max = np.max(a.data)
        expected = np.log(np.sum(np.exp(a.data - a_max))) + a_max

        assert np.isclose(b.data, expected)

        # Test logsumexp with axis
        c = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        d = logsumexp(c, axis=1)

        # Manual calculation with axis
        c_max = np.max(c.data, axis=1, keepdims=True)
        expected = np.log(np.sum(np.exp(c.data - c_max), axis=1)) + c_max.flatten()

        assert np.allclose(d.data, expected)


class TestLossFunctions:
    """Test loss functions."""

    def test_mse_loss(self):
        """Test mean squared error loss."""
        predictions = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        targets = Tensor([[2.0, 2.0], [4.0, 3.0]])

        # Test with mean reduction
        loss = mse_loss(predictions, targets)
        expected = np.mean(np.square(predictions.data - targets.data))
        assert np.isclose(loss.data, expected)

        # Test with sum reduction
        loss_sum = mse_loss(predictions, targets, reduction="sum")
        expected_sum = np.sum(np.square(predictions.data - targets.data))
        assert np.isclose(loss_sum.data, expected_sum)

    def test_binary_cross_entropy(self):
        """Test binary cross entropy loss."""
        predictions = Tensor([0.2, 0.7, 0.9], requires_grad=True)
        targets = Tensor([0.0, 1.0, 1.0])

        # Test with mean reduction
        loss = binary_cross_entropy(predictions, targets)

        # Manual calculation
        p = predictions.data
        t = targets.data
        expected = -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

        assert np.isclose(loss.data, expected)

    def test_softmax_cross_entropy(self):
        """Test softmax cross entropy loss."""
        logits = Tensor(
            [[2.0, 1.0, 0.1], [0.1, 2.0, 1.0], [0.1, 1.0, 2.0]], requires_grad=True
        )
        targets = Tensor([0, 1, 2])  # Class indices

        # Test with mean reduction
        loss = softmax_cross_entropy(logits, targets)

        # Manual calculation
        batch_size = logits.data.shape[0]
        logits_data = logits.data - np.max(logits.data, axis=1, keepdims=True)
        exp_logits = np.exp(logits_data)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        expected = -np.mean(
            np.log([probs[i, targets.data[i]] for i in range(batch_size)])
        )

        assert np.isclose(loss.data, expected)


class TestSimilarityFunctions:
    """Test similarity and distance functions."""

    def test_cosine_similarity(self):
        """Test cosine similarity."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        b = Tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], requires_grad=True)

        sim = cosine_similarity(a, b)

        # Manual calculation
        a_norm = np.sqrt(np.sum(a.data**2, axis=1))
        b_norm = np.sqrt(np.sum(b.data**2, axis=1))
        dot_product = np.sum(a.data * b.data, axis=1)
        expected = dot_product / (a_norm * b_norm)

        assert np.allclose(sim.data, expected)

    def test_pairwise_distance(self):
        """Test pairwise distance."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        b = Tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], requires_grad=True)

        # Test L1 distance
        dist_l1 = pairwise_distance(a, b, p=1.0)
        expected_l1 = np.array(
            [3.0, 9.0]
        )  # |1-1| + |2-1| + |3-1| = 3, |4-2| + |5-2| + |6-2| = 9
        assert np.allclose(dist_l1.data, expected_l1)

        # Test L2 distance
        dist_l2 = pairwise_distance(a, b, p=2.0)
        expected_l2 = np.array(
            [np.sqrt(5.0), np.sqrt(27.0)]
        )  # sqrt((1-1)^2 + (2-1)^2 + (3-1)^2), ...
        assert np.allclose(dist_l2.data, expected_l2)

        # Test backward
        dist_l2.backward(np.array([1.0, 1.0]))

        # Manual gradient computation for L2 distance:
        # grad_a = (a - b) / ||a - b||_2, grad_b = -grad_a
        diff = a.data - b.data
        norm = np.sqrt(np.sum(diff**2, axis=1, keepdims=True))
        expected_grad_a = diff / norm
        expected_grad_b = -expected_grad_a

        assert np.allclose(a.grad, expected_grad_a)
        assert np.allclose(b.grad, expected_grad_b)


class TestAdvancedUseCases:
    """Test advanced use cases combining multiple operations."""

    def test_feedforward_network(self):
        """Test a small feedforward neural network using various ops."""
        # Define network parameters
        W1 = Tensor(np.random.randn(2, 4) * 0.1, requires_grad=True)
        b1 = Tensor(np.zeros(4), requires_grad=True)
        W2 = Tensor(np.random.randn(4, 1) * 0.1, requires_grad=True)
        b2 = Tensor(np.zeros(1), requires_grad=True)

        # Input data
        x = Tensor([[0.5, -0.5], [1.0, -1.0]], requires_grad=True)
        y = Tensor([[0.0], [1.0]])

        # Forward pass
        h1 = matmul(x, W1) + b1
        h1_act = tanh(h1)
        logits = matmul(h1_act, W2) + b2
        probas = sigmoid(logits)
        loss = mse_loss(probas, y)

        # Check that computation graph was built correctly
        assert loss.requires_grad
        assert probas.requires_grad
        assert h1_act.requires_grad
        assert h1.requires_grad

        # Backward pass
        loss.backward()

        # Check that gradients were propagated
        assert W1.grad is not None
        assert b1.grad is not None
        assert W2.grad is not None
        assert b2.grad is not None
        assert x.grad is not None

        # Check gradient shapes
        assert W1.grad.shape == W1.data.shape
        assert b1.grad.shape == b1.data.shape
        assert W2.grad.shape == W2.data.shape
        assert b2.grad.shape == b2.data.shape
        assert x.grad.shape == x.data.shape

    def test_compute_softmax_cross_entropy(self):
        """Test the implementation of softmax cross entropy from scratch."""
        # Create logits and targets
        logits = Tensor([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]], requires_grad=True)
        targets = Tensor([0, 1])  # Class indices

        # Compute softmax probabilities
        # First shift logits for numerical stability
        shifted_logits = logits - logsumexp(logits, axis=1, keepdims=True)
        probas = softmax(logits, axis=1)

        # Create one-hot encoded targets
        batch_size, num_classes = logits.data.shape
        one_hot = zeros((batch_size, num_classes))
        for i in range(batch_size):
            one_hot.data[i, int(targets.data[i])] = 1.0

        # Compute cross entropy loss: -sum(one_hot * log(probas))
        log_probas = shifted_logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -one_hot * log_probas

        # Sum along classes and mean over batch
        loss_sum = loss.sum(axis=1)
        loss_mean = loss_sum.mean()

        # Compare with direct softmax_cross_entropy function
        direct_loss = softmax_cross_entropy(logits, targets)

        # Check that both approaches give the same result
        assert np.isclose(loss_mean.data, direct_loss.data, rtol=1e-5)

        # Check gradients
        loss_mean.backward()
        logits.zero_grad()
        direct_loss.backward()

        # Both approaches should yield the same gradients
        assert np.allclose(logits.grad, probas.data - one_hot.data, rtol=1e-5)
