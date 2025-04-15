from nn.layer import Layer
from core.tensor import Tensor
import numpy as np

class ReLU(Layer):
    def forward(self, x):
        out = Tensor((x.data > 0) * x.data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                grad = (x.data > 0).astype(float)
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out

class Softmax(Layer):
    def forward(self, x: Tensor):
        exps = np.exp(x.data - np.max(x.data, axis=1, keepdims=True))  # for numerical stability
        softmax = exps / np.sum(exps, axis=1, keepdims=True)
        out = Tensor(softmax, requires_grad=x.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if x.requires_grad:
                # Backprop for softmax (simplified, works best with cross-entropy combined)
                grad = out.grad
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out