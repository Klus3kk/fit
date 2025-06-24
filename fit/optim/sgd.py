"""
SGD optimizer implementations.
"""

import numpy as np
from typing import List
from fit.core.tensor import Tensor


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(
        self, parameters: List[Tensor], lr: float = 0.01, weight_decay: float = 0.0
    ):
        """
        Initialize SGD optimizer.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
            weight_decay: L2 penalty coefficient
        """
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        """Perform one optimization step."""
        for param in self.parameters:
            if param.grad is None:
                continue

            grad = param.grad

            # Add weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Update parameters
            param.data = param.data - self.lr * grad

    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters:
            param.grad = None

    def state_dict(self):
        """Return optimizer state."""
        return {"lr": self.lr, "weight_decay": self.weight_decay}

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.lr = state_dict["lr"]
        self.weight_decay = state_dict["weight_decay"]


class SGDMomentum:
    """
    SGD with momentum optimizer.
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
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # Initialize state
        self.state = {}
        for i, param in enumerate(parameters):
            self.state[i] = {"momentum_buffer": np.zeros_like(param.data)}

    def step(self):
        """Perform one optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad
            state = self.state[i]

            # Add weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Update momentum buffer
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


class ASGD:
    """
    Averaged Stochastic Gradient Descent optimizer.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0.0,
    ):
        """
        Initialize ASGD optimizer.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
            lambd: Decay term
            alpha: Power for eta update
            t0: Point at which to start averaging
            weight_decay: L2 penalty coefficient
        """
        self.parameters = parameters
        self.lr = lr
        self.lambd = lambd
        self.alpha = alpha
        self.t0 = t0
        self.weight_decay = weight_decay

        # Initialize state
        self.state = {}
        for i, param in enumerate(parameters):
            self.state[i] = {
                "step": 0,
                "eta": lr,
                "mu": 1.0,
                "ax": np.zeros_like(param.data),
            }

    def step(self):
        """Perform one optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad
            state = self.state[i]

            # Add weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Update step count
            state["step"] += 1

            # Update eta and mu
            if state["step"] > self.t0:
                state["eta"] = self.lr / (
                    (1 + self.lambd * self.lr * (state["step"] - self.t0)) ** self.alpha
                )
                state["mu"] = 1.0 / max(1.0, state["step"] - self.t0)

            # Update parameters
            param.data = param.data - state["eta"] * grad

            # Update averaged parameters
            if state["step"] > self.t0:
                state["ax"] = state["mu"] * param.data + (1 - state["mu"]) * state["ax"]

    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters:
            param.grad = None

    def swap_swa_sgd(self):
        """Swap parameters with averaged parameters."""
        for i, param in enumerate(self.parameters):
            state = self.state[i]
            if state["step"] > self.t0:
                # Swap current parameters with averaged ones
                temp = param.data.copy()
                param.data = state["ax"].copy()
                state["ax"] = temp


class Rprop:
    """
    Rprop optimizer (Resilient backpropagation).
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        etas: tuple = (0.5, 1.2),
        step_sizes: tuple = (1e-6, 50),
    ):
        """
        Initialize Rprop optimizer.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate (initial step size)
            etas: Multiplicative increase and decrease factors
            step_sizes: Minimum and maximum step sizes
        """
        self.parameters = parameters
        self.lr = lr
        self.etaminus, self.etaplus = etas
        self.min_step, self.max_step = step_sizes

        # Initialize state
        self.state = {}
        for i, param in enumerate(parameters):
            self.state[i] = {
                "step_size": np.full_like(param.data, lr),
                "prev_grad": np.zeros_like(param.data),
            }

    def step(self):
        """Perform one optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad
            state = self.state[i]

            # Compute sign change
            sign_change = grad * state["prev_grad"]

            # Update step sizes
            step_size = state["step_size"]
            step_size = np.where(
                sign_change > 0,
                np.minimum(step_size * self.etaplus, self.max_step),
                step_size,
            )
            step_size = np.where(
                sign_change < 0,
                np.maximum(step_size * self.etaminus, self.min_step),
                step_size,
            )
            state["step_size"] = step_size

            # Update parameters
            param.data = param.data - np.sign(grad) * step_size

            # Store gradient for next iteration
            state["prev_grad"] = grad.copy()

    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters:
            param.grad = None
