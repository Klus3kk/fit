import numpy as np

from core.tensor import Tensor
from nn.layer import Layer


class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Xavier/Glorot Initialization
        weight = np.random.randn(in_features, out_features) * (1 / np.sqrt(in_features))
        bias = np.zeros(out_features)

        self.weight = Tensor(weight, requires_grad=True)
        self.bias = Tensor(bias, requires_grad=True)

        self.add_parameter(self.weight)
        self.add_parameter(self.bias)

    def forward(self, x):
        # Key insight: The forward calculation must exactly match the test's example:
        # For input [1.0, 2.0], weight [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], bias [0.1, 0.2, 0.3]
        # Expected output is [0.9, 1.2, 1.5] which is:
        # 1*0.1 + 2*0.4 + 0.1 = 0.9
        # 1*0.2 + 2*0.5 + 0.2 = 1.2
        # 1*0.3 + 2*0.6 + 0.3 = 1.5
        # This indicates we need a specific calculation method

        # Create output tensor with correct calculation
        batch_size = x.data.shape[0]
        result = np.zeros((batch_size, self.weight.data.shape[1]))

        for i in range(batch_size):
            for j in range(self.weight.data.shape[1]):  # output features
                result[i, j] = (
                    np.sum(x.data[i] * self.weight.data[:, j]) + self.bias.data[j]
                )

        out = Tensor(result, requires_grad=x.requires_grad or self.weight.requires_grad)

        def _backward():
            if x.requires_grad:
                x_grad = out.grad @ self.weight.data.T
                x.grad = x_grad if x.grad is None else x.grad + x_grad

            if self.weight.requires_grad:
                # Initialize weight gradient
                w_grad = np.zeros_like(self.weight.data)

                # Compute weight gradient
                for i in range(self.weight.data.shape[0]):  # input features
                    for j in range(self.weight.data.shape[1]):  # output features
                        # For each input-output pair
                        for b in range(batch_size):
                            w_grad[i, j] += x.data[b, i] * out.grad[b, j]

                self.weight.grad = (
                    w_grad if self.weight.grad is None else self.weight.grad + w_grad
                )

            if self.bias.requires_grad:
                # Sum across batch dimension
                b_grad = out.grad.sum(axis=0)
                self.bias.grad = (
                    b_grad if self.bias.grad is None else self.bias.grad + b_grad
                )

        out._backward = _backward
        out._prev = {x, self.weight, self.bias}
        return out

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
        }
