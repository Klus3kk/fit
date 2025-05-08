import numpy as np

from core.tensor import Tensor
from nn.layer import Layer


class ReLU(Layer):
    def forward(self, x):
        out = Tensor((x.data > 0) * x.data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                grad = (x.data > 0).astype(
                    float
                ) * out.grad  # Multiply with upstream grad
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out


class Softmax(Layer):
    def forward(self, x: Tensor):
        exps = np.exp(
            x.data - np.max(x.data, axis=1, keepdims=True)
        )  # for numerical stability
        softmax = exps / np.sum(exps, axis=1, keepdims=True)
        out = Tensor(softmax, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                # Backprop for softmax (simplified, works best with
                # cross-entropy combined)
                grad = out.grad
                x.grad = grad if x.grad is None else x.grad + grad

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
