"""
The key changes:
1. Improved numerical stability in the softmax calculation
2. More robust gradient computation
3. Better handling of target formats (class indices or one-hot)
"""

import numpy as np

from core.tensor import Tensor
from nn.modules.base import Layer


class MSELoss(Layer):
    def forward(self, prediction: Tensor, target: Tensor):
        """
        Compute mean squared error loss manually without using problematic tensor methods.
        """
        # Calculate difference between prediction and target
        diff = prediction - target

        # Compute squared differences
        squared_diff = diff * diff

        # Calculate mean manually
        loss_value = np.mean(squared_diff.data)

        # Create a scalar Tensor with the loss value
        result = Tensor(loss_value, requires_grad=prediction.requires_grad)

        # Define backward pass for gradient calculation
        def _backward():
            if not prediction.requires_grad or result.grad is None:
                return

            # For MSE loss, the gradient is 2 * (prediction - target) / n
            n_elements = np.prod(prediction.data.shape)
            mse_grad = 2.0 * diff.data / n_elements

            # Scale by upstream gradient
            final_grad = mse_grad * result.grad

            # Accumulate gradients
            prediction.grad = (
                final_grad if prediction.grad is None else prediction.grad + final_grad
            )

        # Set up backward function and dependencies
        result._backward = _backward
        result._prev = {prediction}

        return result

