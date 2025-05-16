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
        # Handle edge cases with input shape
        x_data = x.data

        # For single sample input without batch dimension, add a batch dimension
        if x_data.ndim == 1:
            x_data = x_data.reshape(1, -1)

        # Use matrix multiplication for better efficiency
        out_data = np.matmul(x_data, self.weight.data) + self.bias.data

        # Create output tensor with gradients if needed
        out = Tensor(
            out_data, requires_grad=x.requires_grad or self.weight.requires_grad
        )

        # Define backward pass for computing gradients
        def _backward():
            if not out.grad is None:
                # Handle gradient for input tensor
                if x.requires_grad:
                    # x_grad = out.grad @ weight.T
                    if out.grad.ndim < 2:
                        # Handle case when gradient is a scalar or vector
                        reshaped_grad = out.grad.reshape(1, -1)
                        x_grad = np.matmul(reshaped_grad, self.weight.data.T)
                        if x.data.ndim < 2:
                            # If input was 1D, make gradient 1D too
                            x_grad = x_grad.flatten()
                    else:
                        x_grad = np.matmul(out.grad, self.weight.data.T)

                    # Ensure x_grad has the same shape as x.data
                    if x_grad.shape != x.data.shape and x_grad.size == x.data.size:
                        x_grad = x_grad.reshape(x.data.shape)

                    # Accumulate gradients
                    x.grad = x_grad if x.grad is None else x.grad + x_grad

                # Handle gradient for weights
                if self.weight.requires_grad:
                    # For batch inputs: w_grad = x.T @ out.grad
                    if x.data.ndim >= 2:
                        # Handle scalar gradients
                        if out.grad.ndim == 0:
                            grad_reshaped = np.array([out.grad])
                        elif out.grad.ndim == 1:
                            grad_reshaped = out.grad.reshape(-1, 1)
                        else:
                            grad_reshaped = out.grad

                        # Compute weight gradient
                        w_grad = np.matmul(x.data.T, grad_reshaped)
                    else:
                        # For single input: outer product
                        if out.grad.ndim == 0:
                            w_grad = np.outer(x.data, np.array([out.grad]))
                        else:
                            w_grad = np.outer(x.data, out.grad)

                    # Ensure w_grad has the same shape as weight.data
                    if (
                        w_grad.shape != self.weight.data.shape
                        and w_grad.size == self.weight.data.size
                    ):
                        w_grad = w_grad.reshape(self.weight.data.shape)

                    # Accumulate gradients
                    self.weight.grad = (
                        w_grad
                        if self.weight.grad is None
                        else self.weight.grad + w_grad
                    )

                # Handle gradient for bias
                if self.bias.requires_grad:
                    # Sum across batch dimension
                    if out.grad.ndim >= 2:
                        b_grad = np.sum(out.grad, axis=0)
                    else:
                        b_grad = (
                            out.grad.copy() if hasattr(out.grad, "copy") else out.grad
                        )

                    # Ensure b_grad has the same shape as bias.data
                    if (
                        b_grad.shape != self.bias.data.shape
                        and b_grad.size == self.bias.data.size
                    ):
                        b_grad = b_grad.reshape(self.bias.data.shape)

                    # Accumulate gradients
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
