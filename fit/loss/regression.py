"""
Regression loss functions for the FIT framework.
"""

import numpy as np

from core.tensor import Tensor
from nn.modules.base import Layer


class MSELoss(Layer):
    """Mean Squared Error loss function."""

    def forward(self, prediction: Tensor, target: Tensor):
        """
        Compute mean squared error loss.

        Args:
            prediction: Predicted values
            target: Target values

        Returns:
            Tensor containing the MSE loss
        """
        # Ensure target is a tensor (but don't convert if already a Tensor)
        if not isinstance(target, Tensor):
            target = Tensor(target)

        # Calculate difference
        diff = prediction - target

        # Square the differences
        squared_diff = diff * diff

        # Calculate mean manually to avoid issues
        loss_value = np.mean(squared_diff.data)

        # Create result tensor
        result = Tensor(loss_value, requires_grad=prediction.requires_grad)

        if prediction.requires_grad:

            def _backward():
                if result.grad is not None:
                    # Gradient of MSE: 2 * (prediction - target) / n
                    n_elements = np.prod(prediction.data.shape)
                    grad = 2.0 * diff.data / n_elements

                    # Scale by upstream gradient (usually 1.0 for loss)
                    final_grad = grad * result.grad

                    # Accumulate gradients
                    if prediction.grad is None:
                        prediction.grad = final_grad
                    else:
                        prediction.grad = prediction.grad + final_grad

            result._backward = _backward
            result._prev = {prediction}

        return result


class MAELoss(Layer):
    """Mean Absolute Error loss function."""

    def forward(self, prediction: Tensor, target: Tensor):
        """
        Compute mean absolute error loss.

        Args:
            prediction: Predicted values
            target: Target values

        Returns:
            Tensor containing the MAE loss
        """
        # Ensure target is a tensor
        if not isinstance(target, Tensor):
            target = Tensor(target)

        # Calculate difference
        diff = prediction - target

        # Calculate absolute differences manually
        abs_diff_data = np.abs(diff.data)

        # Calculate mean
        loss_value = np.mean(abs_diff_data)

        # Create result tensor
        result = Tensor(loss_value, requires_grad=prediction.requires_grad)

        if prediction.requires_grad:

            def _backward():
                if result.grad is not None:
                    # Gradient of MAE: sign(prediction - target) / n
                    n_elements = np.prod(prediction.data.shape)
                    grad = np.sign(diff.data) / n_elements

                    # Scale by upstream gradient
                    final_grad = grad * result.grad

                    # Accumulate gradients
                    if prediction.grad is None:
                        prediction.grad = final_grad
                    else:
                        prediction.grad = prediction.grad + final_grad

            result._backward = _backward
            result._prev = {prediction}

        return result


class HuberLoss(Layer):
    """Huber loss function (smooth L1 loss)."""

    def __init__(self, delta=1.0):
        """
        Initialize Huber loss.

        Args:
            delta: Threshold parameter
        """
        super().__init__()
        self.delta = delta

    def forward(self, prediction: Tensor, target: Tensor):
        """
        Compute Huber loss.

        Args:
            prediction: Predicted values
            target: Target values

        Returns:
            Tensor containing the Huber loss
        """
        # Ensure target is a tensor
        if not isinstance(target, Tensor):
            target = Tensor(target)

        # Calculate difference
        diff = prediction - target
        abs_diff = np.abs(diff.data)

        # Calculate Huber loss
        mask = abs_diff <= self.delta
        loss_data = np.where(
            mask, 0.5 * diff.data**2, self.delta * abs_diff - 0.5 * self.delta**2
        )

        loss_value = np.mean(loss_data)

        # Create result tensor
        result = Tensor(loss_value, requires_grad=prediction.requires_grad)

        if prediction.requires_grad:

            def _backward():
                if result.grad is not None:
                    # Gradient of Huber loss
                    n_elements = np.prod(prediction.data.shape)
                    grad = (
                        np.where(mask, diff.data, self.delta * np.sign(diff.data))
                        / n_elements
                    )

                    # Scale by upstream gradient
                    final_grad = grad * result.grad

                    # Accumulate gradients
                    if prediction.grad is None:
                        prediction.grad = final_grad
                    else:
                        prediction.grad = prediction.grad + final_grad

            result._backward = _backward
            result._prev = {prediction}

        return result
