import numpy as np

from core.tensor import Tensor
from nn.layer import Layer

class CrossEntropyLoss(Layer):
    """
    Cross Entropy loss for classification tasks.

    This implementation includes the softmax operation,
    so it should be used with raw logits, not softmaxed outputs.
    """

    def forward(self, logits, targets):
        """
        Compute cross entropy loss.

        Args:
            logits: Raw model outputs (before softmax)
            targets: Target class indices or one-hot encoded targets

        Returns:
            Loss tensor
        """
        # Get shapes
        batch_size = logits.data.shape[0]
        num_classes = logits.data.shape[1]

        # Convert targets to class indices if they're one-hot encoded
        if targets.data.ndim > 1 and targets.data.shape[1] > 1:
            target_indices = np.argmax(targets.data, axis=1)
        else:
            target_indices = targets.data.astype(np.int32).reshape(-1)

        # Numerical stability: shift logits to prevent overflow
        logits_shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)

        # Compute softmax: exp(logits) / sum(exp(logits))
        exp_logits = np.exp(logits_shifted)
        softmax_outputs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Select the softmax probabilities for the target classes
        batch_indices = np.arange(batch_size)
        target_probs = softmax_outputs[batch_indices, target_indices]

        # Compute negative log likelihood
        epsilon = 1e-10  # Small constant to avoid log(0)
        nll = -np.log(target_probs + epsilon)

        # Mean over batch
        loss_value = np.mean(nll)

        # Create output tensor
        out = Tensor(loss_value, requires_grad=logits.requires_grad)

        if logits.requires_grad:
            # Store data for backward pass
            def _backward():
                if out.grad is None:
                    return

                # Gradient of cross entropy w.r.t. softmax outputs:
                # -1/N * (one_hot(target) - softmax_output)
                grad = softmax_outputs.copy()
                grad[batch_indices, target_indices] -= 1.0

                # Scale by 1/N and upstream gradient
                grad = grad / batch_size * out.grad

                # Update logits gradient
                logits.grad = grad if logits.grad is None else logits.grad + grad

            # Set up backward function and dependencies
            out._backward = _backward
            out._prev = {logits}

        return out
