"""
Adam optimizer implementation.
"""

import numpy as np
from typing import List
from fit.core.tensor import Tensor


class Adam:
    """
    Adam optimizer with bias correction.

    Combines the advantages of AdaGrad and RMSProp.
    Maintains moving averages of both gradients and squared gradients.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialize Adam optimizer.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages (beta1, beta2)
            eps: Small constant for numerical stability
            weight_decay: L2 penalty coefficient
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize state
        self.state = {}
        for i, param in enumerate(parameters):
            self.state[i] = {
                "m": np.zeros_like(param.data),  # First moment
                "v": np.zeros_like(param.data),  # Second moment
                "step": 0,
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

            # Update biased first moment estimate
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * grad * grad

            # Compute bias-corrected first moment estimate
            m_hat = state["m"] / (1 - self.beta1 ** state["step"])

            # Compute bias-corrected second raw moment estimate
            v_hat = state["v"] / (1 - self.beta2 ** state["step"])

            # Update parameters
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters:
            param.grad = None

    def state_dict(self):
        """Return optimizer state."""
        return {
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "state": self.state,
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.lr = state_dict["lr"]
        self.beta1, self.beta2 = state_dict["betas"]
        self.eps = state_dict["eps"]
        self.weight_decay = state_dict["weight_decay"]
        self.state = state_dict["state"]


class AdamW:
    """
    AdamW optimizer with decoupled weight decay.

    Implements the weight decay fix from "Decoupled Weight Decay Regularization".
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Initialize AdamW optimizer.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages (beta1, beta2)
            eps: Small constant for numerical stability
            weight_decay: Weight decay coefficient (decoupled)
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize state
        self.state = {}
        for i, param in enumerate(parameters):
            self.state[i] = {
                "m": np.zeros_like(param.data),  # First moment
                "v": np.zeros_like(param.data),  # Second moment
                "step": 0,
            }

    def step(self):
        """Perform one optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad
            state = self.state[i]

            # Update step count
            state["step"] += 1

            # Update biased first moment estimate
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * grad * grad

            # Compute bias-corrected first moment estimate
            m_hat = state["m"] / (1 - self.beta1 ** state["step"])

            # Compute bias-corrected second raw moment estimate
            v_hat = state["v"] / (1 - self.beta2 ** state["step"])

            # Update parameters with decoupled weight decay
            param.data = param.data - self.lr * (
                m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param.data
            )

    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters:
            param.grad = None

    def state_dict(self):
        """Return optimizer state."""
        return {
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "state": self.state,
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.lr = state_dict["lr"]
        self.beta1, self.beta2 = state_dict["betas"]
        self.eps = state_dict["eps"]
        self.weight_decay = state_dict["weight_decay"]
        self.state = state_dict["state"]


class Adamax:
    """
    Adamax optimizer (variant of Adam based on infinity norm).
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.002,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialize Adamax optimizer.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages (beta1, beta2)
            eps: Small constant for numerical stability
            weight_decay: L2 penalty coefficient
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize state
        self.state = {}
        for i, param in enumerate(parameters):
            self.state[i] = {
                "m": np.zeros_like(param.data),  # First moment
                "u": np.zeros_like(param.data),  # Infinity norm
                "step": 0,
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

            # Update biased first moment estimate
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad

            # Update infinity norm
            state["u"] = np.maximum(self.beta2 * state["u"], np.abs(grad))

            # Compute bias-corrected first moment estimate
            m_hat = state["m"] / (1 - self.beta1 ** state["step"])

            # Update parameters
            param.data = param.data - self.lr * m_hat / (state["u"] + self.eps)

    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters:
            param.grad = None


class NAdam:
    """
    NAdam optimizer (Nesterov-accelerated Adam).
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.002,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialize NAdam optimizer.

        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages (beta1, beta2)
            eps: Small constant for numerical stability
            weight_decay: L2 penalty coefficient
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize state
        self.state = {}
        for i, param in enumerate(parameters):
            self.state[i] = {
                "m": np.zeros_like(param.data),  # First moment
                "v": np.zeros_like(param.data),  # Second moment
                "step": 0,
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

            # Update biased first moment estimate
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * grad * grad

            # Compute bias-corrected first moment estimate
            m_hat = state["m"] / (1 - self.beta1 ** state["step"])

            # Compute bias-corrected second raw moment estimate
            v_hat = state["v"] / (1 - self.beta2 ** state["step"])

            # Nesterov acceleration
            m_nesterov = self.beta1 * m_hat + (1 - self.beta1) * grad / (
                1 - self.beta1 ** state["step"]
            )

            # Update parameters
            param.data = param.data - self.lr * m_nesterov / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters:
            param.grad = None
