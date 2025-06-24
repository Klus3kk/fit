import numpy as np


class SGD:
    def __init__(self, parameters, lr=0.01, clip_value=5.0):
        self.parameters = parameters
        self.lr = lr
        self.clip_value = clip_value

    def step(self):
        for param in self.parameters:
            if param.grad is None:
                continue  # skip if no gradient

            # Make sure the gradient shape matches parameter shape
            if param.grad.shape != param.data.shape:
                try:
                    # Handle broadcasting
                    if len(param.grad.shape) > len(param.data.shape):
                        axes = tuple(
                            range(len(param.grad.shape) - len(param.data.shape))
                        )
                        param.grad = np.sum(param.grad, axis=axes)

                    # If still no match, try reducing along specific dimensions
                    if param.grad.shape != param.data.shape:
                        reduce_axes = []
                        for i, (grad_dim, param_dim) in enumerate(
                            zip(param.grad.shape, param.data.shape)
                        ):
                            if grad_dim > param_dim:
                                reduce_axes.append(i)

                        if reduce_axes:
                            param.grad = np.sum(
                                param.grad, axis=tuple(reduce_axes), keepdims=True
                            )
                            # Remove extra dimensions
                            param.grad = np.squeeze(param.grad, axis=tuple(reduce_axes))

                    # Last resort
                    if param.grad.shape != param.data.shape:
                        param.grad = param.grad.sum(axis=0)

                except Exception as e:
                    print(
                        f"Cannot align grad shape {param.grad.shape} with param shape {param.data.shape}: {e}"
                    )
                    continue

            # Apply gradient clipping to prevent explosions
            if self.clip_value > 0:
                param.grad = np.clip(param.grad, -self.clip_value, self.clip_value)

            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None


class SGDMomentum:
    def __init__(
        self, parameters, lr=0.01, momentum=0.9, weight_decay=0, clip_value=5.0
    ):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.clip_value = clip_value
        self.velocity = [np.zeros_like(param.data) for param in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            # Make a copy to avoid modifying in-place
            grad = param.grad.copy() if hasattr(param.grad, "copy") else param.grad

            # Handle shape mismatch
            if grad.shape != param.data.shape:
                try:
                    # Handle broadcasting
                    if len(grad.shape) > len(param.data.shape):
                        axes = tuple(range(len(grad.shape) - len(param.data.shape)))
                        grad = np.sum(grad, axis=axes)

                    # If shapes still don't match, try other reductions
                    if grad.shape != param.data.shape:
                        grad = np.sum(grad, axis=0)

                    # If all else fails, try to reshape
                    if grad.shape != param.data.shape and grad.size == param.data.size:
                        grad = grad.reshape(param.data.shape)

                except Exception as e:
                    print(f"Cannot align gradient shape for parameter {i}: {e}")
                    continue

            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data

            # Apply gradient clipping
            if self.clip_value > 0:
                grad = np.clip(grad, -self.clip_value, self.clip_value)

            # Update velocity (momentum)
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grad

            # Update parameters
            param.data = param.data + self.velocity[i]

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None
