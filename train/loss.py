"""
The key changes:
1. Improved numerical stability in the softmax calculation
2. More robust gradient computation
3. Better handling of target formats (class indices or one-hot)
"""

import numpy as np

from core.tensor import Tensor
from nn.layer import Layer


class MSELoss(Layer):
    def forward(self, prediction: Tensor, target: Tensor):
        diff = prediction - target
        return (diff * diff).mean()


class CrossEntropyLoss(Layer):
    def forward(self, logits: Tensor, target: Tensor):
        """
        Compute cross entropy loss.

        Args:
            logits: Raw model outputs (before softmax)
            target: Target class indices

        Returns:
            Loss tensor
        """
        batch_size = logits.data.shape[0]

        # Convert targets to integers if they're not already
        if target.data.ndim > 1 and target.data.shape[1] > 1:
            # One-hot encoded targets
            target_indices = np.argmax(target.data, axis=1)
        else:
            # Class indices
            target_indices = target.data.astype(np.int32).reshape(-1)

        # Apply log-softmax with numerical stability
        # 1. Shift logits for numerical stability
        logits_max = np.max(logits.data, axis=1, keepdims=True)
        logits_shifted = logits.data - logits_max

        # 2. Compute softmax: exp(logits) / sum(exp(logits))
        exp_logits = np.exp(logits_shifted)
        softmax_denominators = np.sum(exp_logits, axis=1, keepdims=True)
        softmax_output = exp_logits / softmax_denominators

        # 3. Compute cross-entropy loss: -log(softmax) for target classes
        batch_indices = np.arange(batch_size)
        target_probs = softmax_output[batch_indices, target_indices]
        losses = -np.log(target_probs + 1e-12)  # Add small epsilon to avoid log(0)
        loss_value = np.mean(losses)

        # Create output tensor
        out = Tensor(loss_value, requires_grad=True)

        # Store for backward pass
        def _backward():
            if logits.requires_grad and out.grad is not None:
                # Initialize gradient as softmax output
                grad = softmax_output.copy()

                # Subtract 1 from the target class positions
                grad[batch_indices, target_indices] -= 1.0

                # Normalize by batch size and multiply by upstream gradient
                grad = grad * (out.grad / batch_size)

                # Accumulate gradient
                logits.grad = grad if logits.grad is None else logits.grad + grad

        out._backward = _backward
        out._prev = {logits}
        return out
