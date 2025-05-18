"""
Enhanced Sharpness-Aware Minimization (SAM) optimizer implementation.

SAM seeks parameters that lie in neighborhoods having uniformly low loss values,
leading to better generalization than standard optimizers.

Paper: https://arxiv.org/abs/2010.01412
"""

import numpy as np
from typing import List, Callable, Dict, Optional, Union

from core.tensor import Tensor


class SAM:
    """
    Implements Sharpness-Aware Minimization (SAM) for improved generalization.

    SAM finds parameters that lie in neighborhoods having uniformly low loss values, 
    making models more robust to perturbations and improving generalization performance.
    This is particularly effective for:
    - Training models that need to generalize well from limited data
    - Improving robustness to adversarial examples
    - Solving problems where sharp minima cause overfitting

    Args:
        parameters: List of parameters to optimize
        base_optimizer: Underlying optimizer (SGD, Adam, etc.) for parameter updates
        rho: Size of the neighborhood to sample for sharpness (default: 0.05)
        epsilon: Small constant for numerical stability (default: 1e-12)
        adaptive: Whether to use adaptive SAM which adjusts influence per parameter (default: False)
    """

    def __init__(
        self, 
        parameters: List[Tensor], 
        base_optimizer: any, 
        rho: float = 0.05, 
        epsilon: float = 1e-12,
        adaptive: bool = False
    ):
        self.parameters = parameters
        self.base_optimizer = base_optimizer  
        self.rho = rho  
        self.epsilon = epsilon  
        self.adaptive = adaptive

        # Store parameter copies for the sharpness-aware update
        self.param_copies = [None for _ in parameters]

    def first_step(self, closure: Callable[[], Tensor]) -> Tensor:
        """
        First step of SAM: Compute gradient at current position, perturb weights
        in the direction of steepest ascent, and save original weights.

        Args:
            closure: A callable that computes the loss and its gradients

        Returns:
            Loss value at the current weights
        """
        # Evaluate loss and compute gradients
        loss = closure()

        # Save current parameter values
        with_grad_params = []
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                with_grad_params.append((i, param))
                self.param_copies[i] = param.data.copy()

        # Skip if no parameters have gradients
        if not with_grad_params:
            return loss

        # Compute norm of the gradients
        grad_norm = self._grad_norm(with_grad_params)
        
        # Skip update if gradient norm is too small
        if grad_norm < self.epsilon or np.isnan(grad_norm):
            return loss

        # Scale factor for the perturbation
        scale = self.rho / (grad_norm + self.epsilon)

        # Add perturbation to the parameters
        for idx, param in with_grad_params:
            if self.adaptive:
                # Adaptive SAM: scale perturbation by parameter norm
                param_norm = np.linalg.norm(param.data)
                adaptive_scale = self.rho * param_norm / (np.linalg.norm(param.grad) + self.epsilon)
                param.data = param.data + adaptive_scale * param.grad
            else:
                # Standard SAM: uniform perturbation scale
                param.data = param.data + scale * param.grad

        # Zero out gradients for the second step
        self.zero_grad()

        return loss

    def second_step(self, closure: Callable[[], Tensor]) -> Tensor:
        """
        Second step of SAM: Compute gradients at the perturbed weights,
        restore original weights, then perform standard optimizer update.

        Args:
            closure: A callable that computes the loss and its gradients

        Returns:
            Loss value at the perturbed weights
        """
        # Compute loss and gradients at the perturbed point
        loss = closure()

        # Restore original weights
        for i, param in enumerate(self.parameters):
            if self.param_copies[i] is not None:
                param.data = self.param_copies[i]
                # Clear the copy to save memory
                self.param_copies[i] = None

        # Perform optimizer update with gradients from perturbed point
        self.base_optimizer.step()

        return loss

    def step(self, closure: Callable[[], Tensor]) -> Tensor:
        """
        Complete SAM update. This performs both steps of SAM in sequence.

        Args:
            closure: A callable that computes the loss and its gradients

        Returns:
            Loss value from the second step
        """
        self.first_step(closure)
        return self.second_step(closure)

    def zero_grad(self) -> None:
        """Zero out gradients of all parameters."""
        self.base_optimizer.zero_grad()

    def _grad_norm(self, with_grad_params: List[tuple]) -> float:
        """
        Compute the norm of gradients.

        Args:
            with_grad_params: List of (index, parameter) tuples having gradients

        Returns:
            Norm of the gradients
        """
        # Compute the square sum of gradients
        norm_sq = 0.0

        for _, param in with_grad_params:
            if param.grad is not None:
                # Square and sum gradients
                grad_sq = np.sum(np.square(param.grad))
                norm_sq += grad_sq

        return np.sqrt(norm_sq)
    
    @property
    def defaults(self) -> Dict[str, any]:
        """Get the default parameters of the optimizer."""
        return {
            "rho": self.rho,
            "epsilon": self.epsilon,
            "adaptive": self.adaptive
        }
    
    def __str__(self) -> str:
        """String representation of the optimizer."""
        return f"SAM(rho={self.rho}, adaptive={self.adaptive}, base_optimizer={self.base_optimizer})"
