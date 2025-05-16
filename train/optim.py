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
                    grad = (
                        grad.sum(axis=0)
                        if grad.shape[0] == param.data.shape[0]
                        else grad.sum(axis=0)
                    )

                except BaseException:
                    raise ValueError(
                        "Cannot align grad shape "
                        + str(grad.shape)
                        + " with param shape "
                        + str(param.data.shape)
                    )

            param.data -= self.lr * grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None


class SGDMomentum:
    def __init__(self, parameters, lr=0.01, momentum=0.9, weight_decay=0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(param.data) for param in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad

            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data

            # Update velocity (momentum)
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grad

            # Update parameters
            param.data = param.data + self.velocity[i]

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None


class Adam:
    def __init__(
        self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    ):
        """
        Initialize Adam optimizer with corrected implementation.
        
        Args:
            parameters: List of parameters to optimize
            lr: Learning rate (default: 0.001)
            betas: Coefficients for running averages (default: (0.9, 0.999))
            eps: Term for numerical stability (default: 1e-8)
            weight_decay: Weight decay factor (default: 0)
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

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

            grad = param.grad.copy()  # Copy to avoid modifying in-place

            # Apply weight decay
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data

            # Update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)

            # Correct bias
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters (with numerical stability)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Clear gradients of all parameters."""
        for param in self.parameters:
            param.grad = None