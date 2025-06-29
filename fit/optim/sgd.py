"""
SGD optimizer implementations.
"""

import numpy as np
from typing import List
from fit.core.tensor import Tensor


class SGD:
    """
    Stochastic Gradient Descent optimizer with optional momentum.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        """
        Initialize SGD optimizer.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
            momentum: Momentum factor (0 for no momentum)
            dampening: Dampening for momentum
            weight_decay: L2 penalty coefficient
            nesterov: Whether to use Nesterov momentum
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # Initialize momentum buffers if momentum > 0
        self.state = {}
        if self.momentum > 0:
            for i, param in enumerate(parameters):
                self.state[i] = {"momentum_buffer": np.zeros_like(param.data)}

    def step(self):
        """Perform one optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad

            # Add weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Apply momentum if specified
            if self.momentum > 0:
                state = self.state[i]
                buf = state["momentum_buffer"]
                buf = self.momentum * buf + (1 - self.dampening) * grad
                state["momentum_buffer"] = buf

                if self.nesterov:
                    # Nesterov momentum
                    grad = grad + self.momentum * buf
                else:
                    # Standard momentum
                    grad = buf

            # Update parameters
            param.data = param.data - self.lr * grad

    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters:
            param.grad = None

    def state_dict(self):
        """Return optimizer state."""
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "dampening": self.dampening,
            "weight_decay": self.weight_decay,
            "nesterov": self.nesterov,
            "state": self.state,
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.lr = state_dict["lr"]
        self.momentum = state_dict["momentum"]
        self.dampening = state_dict["dampening"]
        self.weight_decay = state_dict["weight_decay"]
        self.nesterov = state_dict["nesterov"]
        self.state = state_dict["state"]


class SGDMomentum:
    """
    SGD with momentum optimizer (legacy - kept for compatibility).

    Note: The main SGD class now supports momentum directly.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.9,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        """
        Initialize SGD with momentum optimizer.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
            momentum: Momentum factor
            dampening: Dampening for momentum
            weight_decay: L2 penalty coefficient
            nesterov: Whether to use Nesterov momentum
        """
        # Just delegate to the main SGD class
        self.sgd = SGD(
            parameters=parameters,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

        # Expose the same interface
        self.parameters = self.sgd.parameters
        self.lr = self.sgd.lr
        self.momentum = self.sgd.momentum
        self.dampening = self.sgd.dampening
        self.weight_decay = self.sgd.weight_decay
        self.nesterov = self.sgd.nesterov
        self.state = self.sgd.state

    def step(self):
        """Perform one optimization step."""
        self.sgd.step()

    def zero_grad(self):
        """Zero all parameter gradients."""
        self.sgd.zero_grad()

    def state_dict(self):
        """Return optimizer state."""
        return self.sgd.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.sgd.load_state_dict(state_dict)
