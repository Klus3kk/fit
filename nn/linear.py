import numpy as np

from core.tensor import Tensor
from nn.layer import Layer


class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Store dimensions
        self.in_features = in_features
        self.out_features = out_features

        # Xavier/Glorot Initialization - crucial for training deep networks
        # This keeps the variance of activations and gradients roughly constant
        # across layers, significantly improving convergence
        scale = np.sqrt(2.0 / (in_features + out_features))
        weight = np.random.randn(in_features, out_features) * scale

        # Initialize bias to zeros - generally works better for first iteration
        bias = np.zeros(out_features)

        self.weight = Tensor(weight, requires_grad=True)
        self.bias = Tensor(bias, requires_grad=True)

        self.add_parameter(self.weight)
        self.add_parameter(self.bias)

    def forward(self, x):
        """
        Perform forward pass: y = xW + b

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Use matrix multiplication for better efficiency
        # This helps avoid explicit loops and speeds up computation
        result = np.matmul(x.data, self.weight.data) + self.bias.data

        # Create output tensor with gradients if needed
        out = Tensor(result, requires_grad=x.requires_grad or self.weight.requires_grad)

        # Define backward pass for computing gradients
        def _backward():
            if not out.grad is None:
                # Handle gradient for input tensor
                if x.requires_grad:
                    x_grad = np.matmul(out.grad, self.weight.data.T)
                    x.grad = x_grad if x.grad is None else x.grad + x_grad

                # Handle gradient for weights
                if self.weight.requires_grad:
                    # w_grad = x^T @ out.grad
                    w_grad = np.matmul(x.data.T, out.grad)
                    self.weight.grad = (
                        w_grad
                        if self.weight.grad is None
                        else self.weight.grad + w_grad
                    )

                # Handle gradient for bias
                if self.bias.requires_grad:
                    # Sum across batch dimension
                    b_grad = np.sum(out.grad, axis=0)
                    self.bias.grad = (
                        b_grad if self.bias.grad is None else self.bias.grad + b_grad
                    )

        # Set up backward function and dependencies
        out._backward = _backward
        out._prev = {x, self.weight, self.bias}
        return out

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
        }
