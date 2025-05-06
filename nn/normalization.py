from core.tensor import Tensor
from nn.layer import Layer
import numpy as np


class BatchNorm(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        # Learnable parameters
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        # Add parameters to be tracked
        self.add_parameter(self.gamma)
        self.add_parameter(self.beta)

        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Hyperparameters
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Cache for backward pass
        self.cache = None

    def forward(self, x: Tensor):
        # Reshape for proper broadcasting
        x_data = x.data
        if x_data.ndim == 2:
            # Handle batch of vectors
            x_reshaped = x_data
            reduction_axes = 0
        else:
            # For future support of CNN
            x_reshaped = x_data.reshape(x_data.shape[0], -1)
            reduction_axes = 0

        if self.training:
            # Calculate batch statistics
            batch_mean = np.mean(x_reshaped, axis=reduction_axes, keepdims=True)
            batch_var = np.var(x_reshaped, axis=reduction_axes, keepdims=True)

            # Update running statistics
            self.running_mean = (
                    self.momentum * batch_mean.squeeze() +
                    (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                    self.momentum * batch_var.squeeze() +
                    (1 - self.momentum) * self.running_var
            )

            # Normalize
            x_norm = (x_reshaped - batch_mean) / np.sqrt(batch_var + self.eps)

            # Cache for backward pass
            self.cache = (x_reshaped, batch_mean, batch_var, x_norm)
        else:
            # Use running statistics during inference
            x_norm = (x_reshaped - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # Scale and shift
        out_data = self.gamma.data * x_norm + self.beta.data

        # Reshape back to original shape if needed
        if x_data.ndim != 2:
            out_data = out_data.reshape(x_data.shape)

        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if not x.requires_grad:
                return

            # Get cached values
            x_reshaped, batch_mean, batch_var, x_norm = self.cache

            # Get batch size
            N = x_reshaped.shape[0]

            # Gradient with respect to gamma and beta
            if self.gamma.requires_grad:
                self.gamma.grad = np.sum(out.grad * x_norm, axis=0)

            if self.beta.requires_grad:
                self.beta.grad = np.sum(out.grad, axis=0)

            # Gradient with respect to x
            std_inv = 1.0 / np.sqrt(batch_var + self.eps)

            # Step 1: Gradient through scale and shift
            dx_norm = out.grad * self.gamma.data

            # Step 2: Gradient through normalization
            dx_var = -0.5 * np.sum(dx_norm * (x_reshaped - batch_mean), axis=0) * std_inv ** 3
            dx_mean = -np.sum(dx_norm * std_inv, axis=0) - 2.0 * dx_var * np.mean(x_reshaped - batch_mean, axis=0)

            dx = dx_norm * std_inv + dx_var * 2.0 * (x_reshaped - batch_mean) / N + dx_mean / N

            # Reshape gradient back to original shape if needed
            if x.data.ndim != 2:
                dx = dx.reshape(x.data.shape)

            x.grad = dx if x.grad is None else x.grad + dx

        out._backward = _backward
        out._prev = {x, self.gamma, self.beta}
        return out

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_config(self):
        return {
            "num_features": len(self.gamma.data),
            "eps": self.eps,
            "momentum": self.momentum
        }