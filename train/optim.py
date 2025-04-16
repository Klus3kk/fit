import numpy as np

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is None:
                continue  # skip if no gradient

            grad = param.grad
            if grad.shape != param.data.shape:
                try:
                    # Try reducing dimensions if mismatch
                    grad = grad.sum(axis=0) if grad.shape[0] == param.data.shape[0] else grad.sum(axis=0)
                except:
                    raise ValueError(f"Cannot align grad shape {grad.shape} with param shape {param.data.shape}")

            param.data -= self.lr * grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None
