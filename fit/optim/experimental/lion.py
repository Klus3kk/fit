"""
Lion optimizer implementation.

The Lion (Evolved Sign Momentum) optimizer uses sign-based updates which require less memory
than traditional optimizers like Adam, while often achieving better performance.

Paper: https://arxiv.org/abs/2302.06675
"""

import numpy as np


class Lion:
    """
    Lion optimizer (Evolved Sign Momentum).

    Uses sign-based updates which require less memory than Adam while often
    achieving better performance. Lion typically requires 2-3x larger learning
    rates than Adam.

    Args:
        parameters: List of parameters to optimize
        lr: Learning rate (default: 1e-4)
        betas: Coefficients for computing running averages (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient (default: 0.0)
    """

    def __init__(self, parameters, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay

        # Initialize momentum
        self.m = [np.zeros_like(param.data) for param in parameters]

    def step(self):
        """Performs a single optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # Apply weight decay (directly to gradients)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data

            # Update momentum with current gradient
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Compute update direction (sign-based)
            # Lion uses the sign of a weighted combination of momentum and gradient
            update_direction = np.sign(self.beta2 * self.m[i] + (1 - self.beta2) * grad)

            # Update parameters
            param.data = param.data - self.lr * update_direction

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for param in self.parameters:
            param.grad = None
