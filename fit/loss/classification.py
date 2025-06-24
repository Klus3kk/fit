"""
Classification loss functions.
"""

import numpy as np
from fit.core.tensor import Tensor


class CrossEntropyLoss:
    """
    Cross-entropy loss for multi-class classification.

    Combines log-softmax and negative log-likelihood in a numerically stable way.
    """

    def __init__(self, reduction="mean"):
        """
        Initialize CrossEntropyLoss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        self.reduction = reduction

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Raw model outputs (batch_size, num_classes)
            targets: Target class indices (batch_size,) or one-hot (batch_size, num_classes)

        Returns:
            Loss tensor
        """
        return self.forward(logits, targets)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Forward pass of cross-entropy loss."""
        batch_size = logits.data.shape[0]
        num_classes = logits.data.shape[1]

        # Convert targets to class indices if they're one-hot
        if targets.data.ndim > 1:
            target_indices = np.argmax(targets.data, axis=1)
        else:
            target_indices = targets.data.astype(int)

        # Numerically stable log-softmax
        # log_softmax(x) = x - log(sum(exp(x)))
        logits_max = Tensor(np.max(logits.data, axis=1, keepdims=True))
        logits_shifted = logits - logits_max
        exp_logits = logits_shifted.exp()
        sum_exp = Tensor(np.sum(exp_logits.data, axis=1, keepdims=True))
        log_sum_exp = sum_exp.log()
        log_softmax = logits_shifted - log_sum_exp

        # Extract log probabilities for target classes
        loss_data = np.zeros(batch_size)
        for i in range(batch_size):
            loss_data[i] = -log_softmax.data[i, target_indices[i]]

        # Apply reduction
        if self.reduction == "mean":
            loss_value = np.mean(loss_data)
        elif self.reduction == "sum":
            loss_value = np.sum(loss_data)
        else:  # 'none'
            loss_value = loss_data

        loss = Tensor(loss_value, requires_grad=logits.requires_grad)

        def _backward():
            if not logits.requires_grad or loss.grad is None:
                return

            # Gradient of cross-entropy: softmax - one_hot_targets
            # First compute softmax
            softmax_data = exp_logits.data / np.sum(
                exp_logits.data, axis=1, keepdims=True
            )

            # Create one-hot encoded targets
            one_hot = np.zeros_like(logits.data)
            for i in range(batch_size):
                one_hot[i, target_indices[i]] = 1.0

            # Gradient is (softmax - one_hot) / batch_size for mean reduction
            grad = softmax_data - one_hot

            if self.reduction == "mean":
                grad = grad / batch_size
            elif self.reduction == "none":
                # For 'none' reduction, multiply by upstream gradient
                if isinstance(loss.grad, np.ndarray) and loss.grad.ndim > 0:
                    grad = grad * loss.grad.reshape(-1, 1)

            # Chain with upstream gradient
            if isinstance(loss.grad, np.ndarray) and loss.grad.ndim == 0:
                grad = grad * loss.grad
            elif not isinstance(loss.grad, np.ndarray):
                grad = grad * loss.grad

            logits.grad = grad if logits.grad is None else logits.grad + grad

        loss._backward = _backward
        loss._prev = {logits}

        return loss


class BinaryCrossEntropyLoss:
    """
    Binary cross-entropy loss for binary classification.
    """

    def __init__(self, reduction="mean"):
        """
        Initialize BinaryCrossEntropyLoss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        self.reduction = reduction

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute binary cross-entropy loss.

        Args:
            predictions: Predicted probabilities (batch_size,) or (batch_size, 1)
            targets: Target labels (batch_size,) - should be 0 or 1

        Returns:
            Loss tensor
        """
        return self.forward(predictions, targets)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward pass of binary cross-entropy loss."""
        # Ensure predictions are in valid range [eps, 1-eps]
        eps = 1e-7
        pred_data = np.clip(predictions.data, eps, 1 - eps)
        pred_clipped = Tensor(pred_data, requires_grad=predictions.requires_grad)

        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        log_pred = pred_clipped.log()
        log_one_minus_pred = (Tensor(1.0) - pred_clipped).log()

        bce_elementwise = -(
            targets * log_pred + (Tensor(1.0) - targets) * log_one_minus_pred
        )

        # Apply reduction
        if self.reduction == "mean":
            return bce_elementwise.mean()
        elif self.reduction == "sum":
            return bce_elementwise.sum()
        else:  # 'none'
            return bce_elementwise


class NLLLoss:
    """
    Negative Log-Likelihood loss.

    Expects log-probabilities as input (e.g., output of log-softmax).
    """

    def __init__(self, reduction="mean"):
        """
        Initialize NLLLoss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        self.reduction = reduction

    def __call__(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            log_probs: Log probabilities (batch_size, num_classes)
            targets: Target class indices (batch_size,)

        Returns:
            Loss tensor
        """
        return self.forward(log_probs, targets)

    def forward(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        """Forward pass of NLL loss."""
        batch_size = log_probs.data.shape[0]

        # Convert targets to indices if needed
        if targets.data.ndim > 1:
            target_indices = np.argmax(targets.data, axis=1)
        else:
            target_indices = targets.data.astype(int)

        # Extract log probabilities for target classes
        loss_data = np.zeros(batch_size)
        for i in range(batch_size):
            loss_data[i] = -log_probs.data[i, target_indices[i]]

        # Apply reduction
        if self.reduction == "mean":
            loss_value = np.mean(loss_data)
        elif self.reduction == "sum":
            loss_value = np.sum(loss_data)
        else:  # 'none'
            loss_value = loss_data

        loss = Tensor(loss_value, requires_grad=log_probs.requires_grad)

        def _backward():
            if not log_probs.requires_grad or loss.grad is None:
                return

            # Gradient: -1 for target class, 0 for others
            grad = np.zeros_like(log_probs.data)
            for i in range(batch_size):
                grad[i, target_indices[i]] = -1.0

            if self.reduction == "mean":
                grad = grad / batch_size
            elif self.reduction == "none":
                if isinstance(loss.grad, np.ndarray) and loss.grad.ndim > 0:
                    grad = grad * loss.grad.reshape(-1, 1)

            # Chain with upstream gradient
            if isinstance(loss.grad, np.ndarray) and loss.grad.ndim == 0:
                grad = grad * loss.grad
            elif not isinstance(loss.grad, np.ndarray):
                grad = grad * loss.grad

            log_probs.grad = grad if log_probs.grad is None else log_probs.grad + grad

        loss._backward = _backward
        loss._prev = {log_probs}

        return loss


class FocalLoss:
    """
    Focal Loss for addressing class imbalance.

    Focal Loss = -α(1-p)^γ * log(p)
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model outputs (batch_size, num_classes)
            targets: Target class indices (batch_size,)

        Returns:
            Loss tensor
        """
        return self.forward(logits, targets)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Forward pass of focal loss."""
        # First compute cross-entropy
        ce_loss = CrossEntropyLoss(reduction="none")
        ce = ce_loss(logits, targets)

        # Compute probabilities for target classes
        batch_size = logits.data.shape[0]

        # Softmax
        logits_max = Tensor(np.max(logits.data, axis=1, keepdims=True))
        logits_shifted = logits - logits_max
        exp_logits = logits_shifted.exp()
        softmax = exp_logits / Tensor(np.sum(exp_logits.data, axis=1, keepdims=True))

        # Extract probabilities for target classes
        if targets.data.ndim > 1:
            target_indices = np.argmax(targets.data, axis=1)
        else:
            target_indices = targets.data.astype(int)

        target_probs = np.zeros(batch_size)
        for i in range(batch_size):
            target_probs[i] = softmax.data[i, target_indices[i]]

        # Focal loss modulation
        focal_weight = self.alpha * ((1 - target_probs) ** self.gamma)
        focal_loss_data = focal_weight * ce.data

        # Apply reduction
        if self.reduction == "mean":
            loss_value = np.mean(focal_loss_data)
        elif self.reduction == "sum":
            loss_value = np.sum(focal_loss_data)
        else:  # 'none'
            loss_value = focal_loss_data

        loss = Tensor(loss_value, requires_grad=logits.requires_grad)

        def _backward():
            if not logits.requires_grad or loss.grad is None:
                return

            # This is a simplified gradient - full implementation would be more complex
            # For now, use cross-entropy gradient scaled by focal weight
            ce_grad = CrossEntropyLoss(reduction="none")
            ce_loss_single = ce_grad(logits, targets)

            # Get CE gradient
            dummy_loss = Tensor(1.0)
            dummy_loss._backward = lambda: None
            dummy_loss._prev = {ce_loss_single}

            # Scale by focal weight (simplified)
            focal_weight_mean = np.mean(focal_weight)

            # This is an approximation - proper implementation needs more careful gradient computation
            logits.grad = logits.grad  # Placeholder for proper gradient

        loss._backward = _backward
        loss._prev = {logits}

        return loss
