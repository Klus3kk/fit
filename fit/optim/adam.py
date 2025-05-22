import numpy as np

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
