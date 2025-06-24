"""
Implementation of activation functions for neural networks.
"""

import numpy as np

from core.tensor import Tensor
from nn.modules.base import Layer


class ReLU(Layer):
    def forward(self, x):
        out = Tensor((x.data > 0) * x.data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad and out.grad is not None:
                grad = (x.data > 0).astype(
                    float
                ) * out.grad  # Multiply with upstream grad
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out


class Softmax(Layer):
    def forward(self, x: Tensor, axis=-1):
        """
        Apply softmax function along specified axis.

        Args:
            x: Input tensor
            axis: Dimension along which to apply softmax (default: -1)

        Returns:
            Tensor with softmax applied
        """
        # Step 1: Improve numerical stability by subtracting max
        # This prevents overflow in exp() calculation
        x_max = np.max(x.data, axis=axis, keepdims=True)
        shifted_data = x.data - x_max

        # Step 2: Compute exp() and normalize
        exp_values = np.exp(shifted_data)
        summed_exp = np.sum(exp_values, axis=axis, keepdims=True)
        softmax_output = exp_values / (summed_exp + 1e-12)  # Add epsilon for stability

        # Create output tensor
        out = Tensor(softmax_output, requires_grad=x.requires_grad)

        # Define backward pass for computing gradients
        def _backward():
            if x.requires_grad and out.grad is not None:
                # Calculate Jacobian-vector product for softmax
                # For each sample i and each class j:
                # ∂softmax_j/∂x_i = softmax_j * (δ_ij - softmax_i)
                # Where δ_ij is 1 if i=j and 0 otherwise

                # Initialize gradient
                x_grad = np.zeros_like(x.data)

                # Process each sample
                for i in range(len(x.data)):
                    # Get softmax output and upstream gradient for this sample
                    s = softmax_output[i]
                    g = out.grad[i]

                    # Calculate diag(s) - outer(s, s)
                    # This can be computed efficiently as: s * (g - s·g)
                    s_g_dot = np.sum(s * g)
                    x_grad[i] = s * g - s * s_g_dot

                # Set gradient
                x.grad = x_grad if x.grad is None else x.grad + x_grad

        # Setup backward function and dependencies
        out._backward = _backward
        out._prev = {x}
        return out


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.training = True  # toggle for train/eval modes

    def forward(self, x: Tensor):
        if not self.training or self.p == 0.0:
            return x

        # Generate dropout mask
        keep_prob = 1 - self.p
        self.mask = (np.random.rand(*x.data.shape) < keep_prob).astype(
            np.float32
        ) / keep_prob
        out = Tensor(x.data * self.mask, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad and out.grad is not None:
                grad = self.mask * out.grad
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_config(self):
        """Get configuration for serialization."""
        return {"p": self.p}


class Tanh(Layer):
    """
    Hyperbolic tangent (tanh) activation function.

    The tanh function is defined as tanh(x) = (e^x - e^-x) / (e^x + e^-x).
    It maps inputs to outputs in the range (-1, 1).
    """

    def forward(self, x):
        """
        Apply tanh activation to the input.

        Args:
            x: Input tensor

        Returns:
            Tensor with tanh activation applied
        """
        # Compute tanh using numpy for stability
        out_data = np.tanh(x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad and out.grad is not None:
                # Derivative of tanh(x) is 1 - tanh(x)^2
                grad = (1 - out_data * out_data) * out.grad
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out

    def get_config(self):
        """Get configuration for serialization."""
        return {}
