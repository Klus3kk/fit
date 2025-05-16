"""
Implementation of Sharpness-Aware Minimization (SAM) optimizer.

SAM seeks parameters that lie in neighborhoods having uniformly low loss value,
leading to better generalization than standard optimizers.

Paper: https://arxiv.org/abs/2010.01412
"""

import numpy as np

from core.tensor import Tensor


class SAM:
    """
    Implements Sharpness-Aware Minimization for improved generalization.

    This optimizer finds parameters that lie in neighborhoods having uniformly
    low loss, leading to better generalization than standard methods.

    Args:
        parameters: List of parameters to optimize
        base_optimizer: Underlying optimizer (like SGD, Adam) for updating parameters
        rho: Size of the neighborhood to sample for sharpness (default: 0.05)
        epsilon: Small constant for numerical stability (default: 1e-12)
    """

    def __init__(self, parameters, base_optimizer, rho=0.05, epsilon=1e-12):
        self.parameters = parameters
        self.base_optimizer = base_optimizer  # Underlying optimizer (SGD, Adam, etc.)
        self.rho = rho  # Neighborhood size for sharpness calculation
        self.epsilon = epsilon  # For numerical stability

        # Store parameter copies for the sharpness-aware update
        self.param_copies = [None for _ in parameters]

    def first_step(self, closure):
        """
        First step of SAM: Compute gradient, perturb weights in sharpest direction,
        and save original weights.

        Args:
            closure: A closure that re-evaluates the model and returns the loss
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

        # Compute and normalize the gradient norm
        grad_norm = self._grad_norm(with_grad_params)

        # Skip update if gradient norm is too small
        if grad_norm < self.epsilon or np.isnan(grad_norm):
            return loss

        scale = self.rho / (grad_norm + self.epsilon)

        # Add perturbation to the parameters by moving in the direction
        # of greatest sharpness (the gradient direction)
        for idx, param in with_grad_params:
            # Make sure gradients have compatible shapes
            if param.grad.shape != param.data.shape:
                try:
                    # Handle broadcasting
                    if len(param.grad.shape) > len(param.data.shape):
                        axes = tuple(
                            range(len(param.grad.shape) - len(param.data.shape))
                        )
                        grad = np.sum(param.grad, axis=axes)
                    else:
                        grad = (
                            param.grad.copy()
                            if hasattr(param.grad, "copy")
                            else param.grad
                        )

                    # Reshape if needed and possible
                    if grad.shape != param.data.shape and grad.size == param.data.size:
                        grad = grad.reshape(param.data.shape)

                    # If shapes still don't match, skip
                    if grad.shape != param.data.shape:
                        continue

                    # Apply perturbation
                    param.data = param.data + scale * grad

                except Exception as e:
                    print(f"Error in SAM first step: {e}")
                    continue
            else:
                # Shapes match, apply perturbation directly
                param.data = param.data + scale * param.grad

        # Zero out gradients for the second step
        self.zero_grad()

        return loss

    def second_step(self, closure):
        """
        Second step of SAM: Compute gradients at the perturbed weights,
        restore original weights, then do a standard optimizer update.

        Args:
            closure: A closure that re-evaluates the model and returns the loss
        """
        # Compute loss and gradients at the perturbed point
        loss = closure()

        # Restore original weights but keep gradients from the perturbed point
        for i, param in enumerate(self.parameters):
            if self.param_copies[i] is not None:
                param.data = self.param_copies[i]

        # Perform the optimizer update using base_optimizer
        self.base_optimizer.step()

        return loss

    def step(self, closure):
        """
        Complete SAM update. This performs both steps of SAM.

        Args:
            closure: A closure that re-evaluates the model and returns the loss
        """
        loss = self.first_step(closure)
        return self.second_step(closure)

    def zero_grad(self):
        """Zero out gradients of all parameters."""
        self.base_optimizer.zero_grad()

    def _grad_norm(self, with_grad_params):
        """
        Compute the norm of gradients.

        Args:
            with_grad_params: List of (index, parameter) tuples having gradients

        Returns:
            Norm of the gradients
        """
        # Compute the square sum of gradients
        norm = 0.0

        for _, param in with_grad_params:
            if param.grad is not None:
                # Square and sum gradients
                norm += np.sum(np.square(param.grad))

        return np.sqrt(norm)
