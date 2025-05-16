"""
Implementation of activation functions for neural networks.
"""

import numpy as np

from core.tensor import Tensor
from nn.layer import Layer



class ReLU(Layer):
    def forward(self, x):
        out = Tensor((x.data > 0) * x.data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                grad = (x.data > 0).astype(float) * out.grad  # Multiply with upstream grad
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
                # The Jacobian of softmax is complex
                # For each i, j: ∂S_i/∂x_j = S_i * (δ_ij - S_j)
                # We need to handle this efficiently
                
                if axis == -1 or axis == x.data.ndim - 1:
                    # Efficient vectorized implementation for the common case
                    # where softmax is applied along the last dimension
                    
                    # Compute S_i * grad_i for each sample in batch
                    si_gi = softmax_output * out.grad
                    
                    # Compute S_i * sum(S_j * grad_j) for each sample
                    sum_sj_gj = np.sum(si_gi, axis=axis, keepdims=True)
                    
                    # Final gradient: S_i * (grad_i - sum(S_j * grad_j))
                    dx = softmax_output * (out.grad - sum_sj_gj)
                    
                    # Accumulate gradient
                    x.grad = dx if x.grad is None else x.grad + dx
                else:
                    # For other axes, we use a more general approach
                    # This is less efficient but handles all cases
                    dx = np.zeros_like(x.data)
                    
                    # For simplicity and clarity, we'll use a very basic implementation
                    # A production version would optimize this further
                    for i in range(len(x.data)):
                        s = softmax_output[i]
                        g = out.grad[i]
                        
                        # Diagonal term: S_i * (1 - S_i) * grad_i
                        # Off-diagonal term: -S_i * S_j * grad_j
                        dx[i] = s * g - s * np.sum(s * g, axis=axis, keepdims=True)
                    
                    # Accumulate gradient
                    x.grad = dx if x.grad is None else x.grad + dx

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
        self.mask = (np.random.rand(*x.data.shape) < keep_prob).astype(np.float32) / keep_prob
        out = Tensor(x.data * self.mask, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
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
        # Compute tanh
        out_data = np.tanh(x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                # Derivative of tanh(x) is 1 - tanh(x)^2
                grad = (1 - out_data * out_data) * out.grad
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out

    def get_config(self):
        """Get configuration for serialization."""
        return {}
