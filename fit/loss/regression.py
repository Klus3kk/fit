"""
Regression loss functions.
"""

import numpy as np
from fit.core.tensor import Tensor


class MSELoss:
    """
    Mean Squared Error loss for regression.

    L(y, ŷ) = (1/n) * Σ(y - ŷ)²
    """

    def __init__(self, reduction="mean"):
        """
        Initialize MSELoss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        self.reduction = reduction

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute mean squared error loss.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            Loss tensor
        """
        return self.forward(predictions, targets)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward pass of MSE loss."""
        # Compute squared differences
        diff = predictions - targets
        squared_diff = diff * diff

        # Apply reduction
        if self.reduction == "mean":
            return squared_diff.mean()
        elif self.reduction == "sum":
            return squared_diff.sum()
        else:  # 'none'
            return squared_diff


class MAELoss:
    """
    Mean Absolute Error loss for regression.

    L(y, ŷ) = (1/n) * Σ|y - ŷ|
    """

    def __init__(self, reduction="mean"):
        """
        Initialize MAELoss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        self.reduction = reduction

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute mean absolute error loss.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            Loss tensor
        """
        return self.forward(predictions, targets)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward pass of MAE loss."""
        # Compute absolute differences
        diff = predictions - targets
        abs_diff = Tensor(np.abs(diff.data), requires_grad=diff.requires_grad)

        def _backward():
            if diff.requires_grad and abs_diff.grad is not None:
                # Gradient of abs(x) is sign(x)
                grad = abs_diff.grad * np.sign(diff.data)
                diff.grad = grad if diff.grad is None else diff.grad + grad

        abs_diff._backward = _backward
        abs_diff._prev = {diff}

        # Apply reduction
        if self.reduction == "mean":
            return abs_diff.mean()
        elif self.reduction == "sum":
            return abs_diff.sum()
        else:  # 'none'
            return abs_diff


class SmoothL1Loss:
    """
    Smooth L1 Loss (Huber Loss) for regression.

    Less sensitive to outliers than MSE.
    L(x) = 0.5*x² if |x| < β
           β*|x| - 0.5*β² otherwise
    """

    def __init__(self, beta=1.0, reduction="mean"):
        """
        Initialize SmoothL1Loss.

        Args:
            beta: Threshold for switching between L2 and L1 loss
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        self.beta = beta
        self.reduction = reduction

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute smooth L1 loss.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            Loss tensor
        """
        return self.forward(predictions, targets)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward pass of Smooth L1 loss."""
        diff = predictions - targets
        abs_diff = np.abs(diff.data)

        # Smooth L1 loss
        loss_data = np.where(
            abs_diff < self.beta,
            0.5 * diff.data * diff.data / self.beta,
            abs_diff - 0.5 * self.beta,
        )

        loss_tensor = Tensor(loss_data, requires_grad=diff.requires_grad)

        def _backward():
            if diff.requires_grad and loss_tensor.grad is not None:
                # Gradient computation
                grad = np.where(
                    abs_diff < self.beta, diff.data / self.beta, np.sign(diff.data)
                )
                grad = grad * loss_tensor.grad
                diff.grad = grad if diff.grad is None else diff.grad + grad

        loss_tensor._backward = _backward
        loss_tensor._prev = {diff}

        # Apply reduction
        if self.reduction == "mean":
            return loss_tensor.mean()
        elif self.reduction == "sum":
            return loss_tensor.sum()
        else:  # 'none'
            return loss_tensor


class HuberLoss:
    """
    Huber Loss for regression.

    Combines MSE and MAE - quadratic for small errors, linear for large errors.
    """

    def __init__(self, delta=1.0, reduction="mean"):
        """
        Initialize Huber Loss.

        Args:
            delta: Threshold for switching between quadratic and linear
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        self.delta = delta
        self.reduction = reduction

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute Huber loss.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            Loss tensor
        """
        return self.forward(predictions, targets)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward pass of Huber loss."""
        diff = predictions - targets
        abs_diff = np.abs(diff.data)

        # Huber loss
        loss_data = np.where(
            abs_diff <= self.delta,
            0.5 * diff.data * diff.data,
            self.delta * abs_diff - 0.5 * self.delta * self.delta,
        )

        loss_tensor = Tensor(loss_data, requires_grad=diff.requires_grad)

        def _backward():
            if diff.requires_grad and loss_tensor.grad is not None:
                # Gradient computation
                grad = np.where(
                    abs_diff <= self.delta, diff.data, self.delta * np.sign(diff.data)
                )
                grad = grad * loss_tensor.grad
                diff.grad = grad if diff.grad is None else diff.grad + grad

        loss_tensor._backward = _backward
        loss_tensor._prev = {diff}

        # Apply reduction
        if self.reduction == "mean":
            return loss_tensor.mean()
        elif self.reduction == "sum":
            return loss_tensor.sum()
        else:  # 'none'
            return loss_tensor


class LogCoshLoss:
    """
    Logarithm of the hyperbolic cosine loss.

    Smooth approximation to MAE that is less sensitive to outliers than MSE.
    """

    def __init__(self, reduction="mean"):
        """
        Initialize LogCosh Loss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        self.reduction = reduction

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute log-cosh loss.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            Loss tensor
        """
        return self.forward(predictions, targets)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward pass of log-cosh loss."""
        diff = predictions - targets

        # log(cosh(x)) = log((e^x + e^-x)/2) = log(e^x + e^-x) - log(2)
        # For numerical stability, use: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
        abs_diff = np.abs(diff.data)
        loss_data = abs_diff + np.log(1 + np.exp(-2 * abs_diff)) - np.log(2)

        loss_tensor = Tensor(loss_data, requires_grad=diff.requires_grad)

        def _backward():
            if diff.requires_grad and loss_tensor.grad is not None:
                # Gradient of log(cosh(x)) is tanh(x)
                grad = np.tanh(diff.data) * loss_tensor.grad
                diff.grad = grad if diff.grad is None else diff.grad + grad

        loss_tensor._backward = _backward
        loss_tensor._prev = {diff}

        # Apply reduction
        if self.reduction == "mean":
            return loss_tensor.mean()
        elif self.reduction == "sum":
            return loss_tensor.sum()
        else:  # 'none'
            return loss_tensor


class QuantileLoss:
    """
    Quantile Loss for quantile regression.

    Allows predicting specific quantiles of the target distribution.
    """

    def __init__(self, quantile=0.5, reduction="mean"):
        """
        Initialize Quantile Loss.

        Args:
            quantile: Quantile to predict (0 < quantile < 1)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        if not 0 < quantile < 1:
            raise ValueError("Quantile must be between 0 and 1")
        self.quantile = quantile
        self.reduction = reduction

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute quantile loss.

        Args:
            predictions: Predicted quantile values
            targets: Target values

        Returns:
            Loss tensor
        """
        return self.forward(predictions, targets)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward pass of quantile loss."""
        diff = targets - predictions

        # Quantile loss: max(q*diff, (q-1)*diff)
        loss_data = np.maximum(
            self.quantile * diff.data, (self.quantile - 1) * diff.data
        )

        loss_tensor = Tensor(loss_data, requires_grad=diff.requires_grad)

        def _backward():
            if diff.requires_grad and loss_tensor.grad is not None:
                # Gradient is q if diff > 0, else (q-1)
                grad = np.where(diff.data > 0, self.quantile, self.quantile - 1)
                # Note: gradient w.r.t. predictions is -grad
                pred_grad = -grad * loss_tensor.grad
                predictions.grad = (
                    pred_grad
                    if predictions.grad is None
                    else predictions.grad + pred_grad
                )

        loss_tensor._backward = _backward
        loss_tensor._prev = {predictions}

        # Apply reduction
        if self.reduction == "mean":
            return loss_tensor.mean()
        elif self.reduction == "sum":
            return loss_tensor.sum()
        else:  # 'none'
            return loss_tensor


class CosineSimilarityLoss:
    """
    Cosine Similarity Loss for regression.

    Measures the cosine of the angle between prediction and target vectors.
    """

    def __init__(self, reduction="mean"):
        """
        Initialize Cosine Similarity Loss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        self.reduction = reduction

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute cosine similarity loss.

        Args:
            predictions: Predicted vectors
            targets: Target vectors

        Returns:
            Loss tensor (1 - cosine_similarity)
        """
        return self.forward(predictions, targets)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward pass of cosine similarity loss."""
        # Cosine similarity = (A·B) / (||A|| * ||B||)
        dot_product = (predictions * targets).sum(axis=-1, keepdims=True)

        pred_norm = (predictions * predictions).sum(axis=-1, keepdims=True).sqrt()
        target_norm = (targets * targets).sum(axis=-1, keepdims=True).sqrt()

        # Add small epsilon for numerical stability
        eps = 1e-8
        pred_norm = pred_norm + Tensor(eps)
        target_norm = target_norm + Tensor(eps)

        cosine_sim = dot_product / (pred_norm * target_norm)

        # Loss is 1 - cosine_similarity
        loss = Tensor(1.0) - cosine_sim

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
