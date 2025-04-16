from core.tensor import Tensor
from nn.layer import Layer
import numpy as np

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
        out = x @ self.weight + self.bias

        def _backward():
            if x.requires_grad:
                x_grad = out.grad @ self.weight.data.T
                x.grad = x_grad if x.grad is None else x.grad + x_grad
            if self.weight.requires_grad:
                w_grad = x.data.T @ out.grad
                self.weight.grad = w_grad if self.weight.grad is None else self.weight.grad + w_grad
            if self.bias.requires_grad:
                b_grad = out.grad.sum(axis=0)
                self.bias.grad = b_grad if self.bias.grad is None else self.bias.grad + b_grad

        out._backward = _backward
        out._prev = {x, self.weight, self.bias}
        return out


