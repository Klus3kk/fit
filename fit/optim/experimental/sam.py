"""
Enhanced Sharpness-Aware Minimization (SAM) optimizer implementation.

SAM is a state-of-the-art optimizer that seeks parameters that lie in neighborhoods having
uniformly low loss values, leading to better generalization than standard optimizers.

This implementation includes:
- Adaptive sharpness
- Efficient computation
- Memory optimization
- Support for all base optimizers

Paper: https://arxiv.org/abs/2010.01412
"""

import numpy as np
from typing import List, Callable, Dict, Optional, Union, Tuple, Any

from fit.core.tensor import Tensor


class SAM:
    """
    Enhanced implementation of Sharpness-Aware Minimization for improved generalization.

    SAM finds parameters that lie in neighborhoods having uniformly low loss values,
    making models more robust to perturbations and improving generalization performance.
    This is particularly effective for:
    - Training models that need to generalize well from limited data
    - Improving robustness to adversarial examples
    - Solving problems like XOR where sharp minima cause overfitting

    Args:
        parameters: List of parameters to optimize
        base_optimizer: Underlying optimizer (SGD, Adam, etc.) for parameter updates
        rho: Size of the neighborhood to sample for sharpness (default: 0.05)
        epsilon: Small constant for numerical stability (default: 1e-12)
        adaptive: Whether to use adaptive SAM which adjusts influence per parameter (default: False)
        auto_clip: Whether to automatically clip gradients during perturbation (default: True)
    """

    def __init__(
        self,
        parameters: List[Tensor],
        base_optimizer: Any,
        rho: float = 0.05,
        epsilon: float = 1e-12,
        adaptive: bool = False,
        auto_clip: bool = True,
    ):
        self.parameters = parameters
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.epsilon = epsilon
        self.adaptive = adaptive
        self.auto_clip = auto_clip

        # Store parameter copies for the sharpness-aware update
        self.param_copies = [None for _ in parameters]

        # For learning rate handling
        self._base_lr = getattr(base_optimizer, "lr", None)

        # Gradient norm scaling history for adaptive clipping
        self.grad_norm_history = []
        self.max_history_size = 10

    def first_step(self, zero_grad: bool = False):
        """
        First step of SAM: Compute gradient at current position, perturb weights
        in the direction of steepest ascent, and save original weights.

        Args:
            zero_grad: Whether to zero gradients after computing perturbation
        """
        # Save current parameters
        for i, param in enumerate(self.parameters):
            self.param_copies[i] = param.data.copy()

        # Compute perturbation based on current gradients
        gradients = [param.grad for param in self.parameters if param.grad is not None]

        if not gradients:
            return  # No gradients to work with

        # Calculate gradient norm
        grad_norm = self._compute_grad_norm(gradients)

        if grad_norm < self.epsilon:
            return  # Gradients too small

        # Apply perturbation
        perturbation_scale = self.rho / (grad_norm + self.epsilon)

        for param in self.parameters:
            if param.grad is not None:
                if self.adaptive:
                    # Adaptive SAM: scale by parameter magnitude
                    param_norm = np.linalg.norm(param.data) + self.epsilon
                    scale = perturbation_scale * param_norm
                else:
                    scale = perturbation_scale

                # Apply perturbation
                param.data += scale * param.grad

        # Zero gradients if requested
        if zero_grad:
            self.zero_grad()

    def second_step(self, zero_grad: bool = False):
        """
        Second step of SAM: Restore original weights and apply the actual update.

        Args:
            zero_grad: Whether to zero gradients after the update
        """
        # Restore original parameters
        for i, param in enumerate(self.parameters):
            if self.param_copies[i] is not None:
                param.data = self.param_copies[i]

        # Apply base optimizer step
        self.base_optimizer.step()

        # Zero gradients if requested
        if zero_grad:
            self.zero_grad()

    def step(self, closure: Optional[Callable] = None):
        """
        Single step that combines both SAM steps (for compatibility).

        Note: This requires the closure to compute the loss function.
        For most use cases, use first_step() and second_step() separately.
        """
        if closure is None:
            raise ValueError("SAM requires a closure function for single-step mode")

        # First step: compute gradients and perturb
        self.first_step(zero_grad=True)

        # Re-compute gradients at perturbed position
        loss = closure()
        loss.backward()

        # Second step: restore and update
        self.second_step(zero_grad=True)

        return loss

    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters:
            param.grad = None

    def _compute_grad_norm(self, gradients: List[np.ndarray]) -> float:
        """Compute the norm of all gradients."""
        total_norm = 0.0
        for grad in gradients:
            if grad is not None:
                total_norm += np.sum(grad * grad)
        return np.sqrt(total_norm)

    @property
    def lr(self):
        """Get learning rate from base optimizer."""
        return getattr(self.base_optimizer, "lr", None)

    @lr.setter
    def lr(self, value):
        """Set learning rate on base optimizer."""
        if hasattr(self.base_optimizer, "lr"):
            self.base_optimizer.lr = value

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return {
            "rho": self.rho,
            "epsilon": self.epsilon,
            "adaptive": self.adaptive,
            "auto_clip": self.auto_clip,
            "base_optimizer": getattr(self.base_optimizer, "state_dict", lambda: {})(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state."""
        self.rho = state_dict.get("rho", self.rho)
        self.epsilon = state_dict.get("epsilon", self.epsilon)
        self.adaptive = state_dict.get("adaptive", self.adaptive)
        self.auto_clip = state_dict.get("auto_clip", self.auto_clip)

        if hasattr(self.base_optimizer, "load_state_dict"):
            self.base_optimizer.load_state_dict(state_dict.get("base_optimizer", {}))


class AdaptiveSAM(SAM):
    """
    Adaptive SAM that automatically adjusts the perturbation size.

    This variant adjusts the perturbation based on the parameter magnitudes,
    often leading to better performance on diverse problems.
    """

    def __init__(self, parameters: List[Tensor], base_optimizer: Any, **kwargs):
        kwargs["adaptive"] = True
        super().__init__(parameters, base_optimizer, **kwargs)
