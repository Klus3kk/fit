"""
Test the Tanh activation function.
"""

import numpy as np
import pytest

from core.tensor import Tensor
from nn.activations import Tanh


class TestTanh:
    def test_tanh_forward(self):
        """Test forward pass of Tanh activation."""
        tanh = Tanh()
        x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        out = tanh(x)

        # Expected values: tanh(-2) ≈ -0.964, tanh(-1) ≈ -0.762,
        # tanh(0) = 0, tanh(1) ≈ 0.762, tanh(2) ≈ 0.964
        expected = np.tanh(x.data)

        assert np.allclose(out.data, expected)

    def test_tanh_backward(self):
        """Test backward pass of Tanh activation."""
        tanh = Tanh()
        x = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
        out = tanh(x)

        # Set upstream gradient
        out.grad = np.array([1.0, 1.0, 1.0])
        out._backward()

        # Derivative of tanh(x) is 1 - tanh(x)^2
        expected_grad = 1 - np.tanh(x.data) ** 2

        assert np.allclose(x.grad, expected_grad)
