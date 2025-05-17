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


class Adam:
    def __init__(
        self,
        parameters,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        clip_value=5.0,
    ):
        """
        Initialize Adam optimizer with robust implementation.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate (default: 0.001)
            betas: Coefficients for running averages (default: (0.9, 0.999))
            eps: Term for numerical stability (default: 1e-8)
            weight_decay: Weight decay factor (default: 0)
            clip_value: Maximum gradient value for clipping (default: 5.0)
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_value = clip_value

        # Initialize momentum and velocity
        self.m = [np.zeros_like(param.data) for param in parameters]
        self.v = [np.zeros_like(param.data) for param in parameters]
        self.t = 0

        # Validate parameters
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

    def step(self):
        self.t += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            # Copy to avoid modifying in-place
            grad = param.grad.copy() if hasattr(param.grad, "copy") else param.grad

            # Handle shape mismatch if needed
            if grad.shape != param.data.shape:
                try:
                    # Handle broadcasting
                    if len(grad.shape) > len(param.data.shape):
                        axes = tuple(range(len(grad.shape) - len(param.data.shape)))
                        grad = np.sum(grad, axis=axes)

                    # If shapes still don't match, try other dimension reductions
                    if grad.shape != param.data.shape:
                        grad = np.sum(grad, axis=0)

                    # Reshape if needed
                    if grad.shape != param.data.shape:
                        # Last resort: try to reshape (if same number of elements)
                        if grad.size == param.data.size:
                            grad = grad.reshape(param.data.shape)
                        else:
                            # Skip this parameter
                            continue

                except Exception as e:
                    print(f"Cannot align gradient shape for parameter {i}: {e}")
                    continue

            # Apply weight decay
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data

            # Apply gradient clipping to prevent explosion
            if self.clip_value > 0:
                grad = np.clip(grad, -self.clip_value, self.clip_value)

            # Update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)

            # Correct bias
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update parameters (with numerical stability)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Clear gradients of all parameters."""
        for param in self.parameters:
            param.grad = None
