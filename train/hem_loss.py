"""
Implementation of High Error Margin (HEM) loss based on the paper
"A margin-based replacement for cross-entropy loss" by Spratling and Schütt.

This loss provides better generalization and robustness compared to standard
cross-entropy loss, while maintaining good classification performance.
"""

import numpy as np
from core.tensor import Tensor

class HEMLoss:
    def __init__(self, margin=0.5):
        """
        Initialize HEM loss with specified margin.
        
        Args:
            margin: Margin parameter μ that controls separation between classes
        """
        self.margin = margin
    
    def __call__(self, logits: Tensor, targets: Tensor):
        """
        Compute HEM loss for binary or multi-class classification.
        
        Args:
            logits: Raw output from the network (pre-softmax)
            targets: Ground truth class indices or values
        
        Returns:
            Loss tensor
        """
        batch_size = logits.data.shape[0]
        
        # Handle different shapes of logits
        if logits.data.ndim == 2 and logits.data.shape[1] == 1:
            # Binary classification with single output neuron
            return self._binary_loss(logits, targets)
        else:
            # Multi-class classification
            return self._multiclass_loss(logits, targets)
    
    def _binary_loss(self, logits: Tensor, targets: Tensor):
        """
        Compute HEM loss for binary classification.
        
        Args:
            logits: Raw output from the network (shape: batch_size x 1)
            targets: Ground truth values (0 or 1)
            
        Returns:
            Loss tensor
        """
        batch_size = logits.data.shape[0]
        
        # Convert targets to flat array if needed
        if targets.data.ndim > 1:
            target_values = targets.data.flatten()
        else:
            target_values = targets.data
            
        # Compute errors for each sample
        errors = np.zeros((batch_size, 2))  # Two errors: one for each class
        
        for i in range(batch_size):
            pred = logits.data[i, 0]
            target = target_values[i]
            
            # If target is 1, the prediction should be high (positive)
            # If target is 0, the prediction should be low (negative)
            if target > 0.5:  # Target is 1
                # Error if prediction is too low
                errors[i, 0] = max(0, self.margin - pred)
            else:  # Target is 0
                # Error if prediction is too high
                errors[i, 1] = max(0, pred + self.margin)
                
        # For each sample, threshold errors by mean and compute mean of above-zero values
        sample_losses = []
        for i in range(batch_size):
            # Get non-zero errors
            non_zero_errors = errors[i][errors[i] > 0]
            if len(non_zero_errors) == 0:
                sample_losses.append(0.0)
                continue
                
            # Calculate mean error
            mean_error = np.mean(non_zero_errors)
            
            # Set errors below mean to zero
            above_mean_errors = non_zero_errors[non_zero_errors >= mean_error]
            
            if len(above_mean_errors) > 0:
                # Calculate mean of above-mean errors
                sample_losses.append(np.mean(above_mean_errors))
            else:
                sample_losses.append(0.0)
        
        # Calculate final loss (mean of all sample losses)
        total_loss = np.mean(sample_losses)
        
        # Create output tensor
        out = Tensor(total_loss, requires_grad=True)
        
        def _backward():
            if logits.requires_grad and out.grad is not None:
                # Initialize gradient
                grad = np.zeros_like(logits.data)
                
                for i in range(batch_size):
                    pred = logits.data[i, 0]
                    target = target_values[i]
                    
                    # Calculate sample errors
                    sample_errors = errors[i]
                    non_zero_errors = sample_errors[sample_errors > 0]
                    
                    if len(non_zero_errors) == 0:
                        continue
                        
                    # Calculate mean error
                    mean_error = np.mean(non_zero_errors)
                    
                    # Count errors above mean
                    above_mean_mask = sample_errors >= mean_error
                    num_above_mean = np.sum(above_mean_mask)
                    
                    if num_above_mean == 0:
                        continue
                    
                    # Calculate gradient based on which error(s) are above mean
                    if target > 0.5:  # Target is 1
                        if above_mean_mask[0]:  # Error for target 1 is above mean
                            grad[i, 0] -= 1.0 / num_above_mean  # Increase prediction
                    else:  # Target is 0
                        if above_mean_mask[1]:  # Error for target 0 is above mean
                            grad[i, 0] += 1.0 / num_above_mean  # Decrease prediction
                
                # Apply upstream gradient
                grad = grad * (out.grad / batch_size)
                
                # Accumulate gradient
                logits.grad = grad if logits.grad is None else logits.grad + grad
        
        out._backward = _backward
        out._prev = {logits}
        return out
        
    def _multiclass_loss(self, logits: Tensor, targets: Tensor):
        """
        Compute HEM loss for multi-class classification.
        
        Args:
            logits: Raw output from the network (pre-softmax)
            targets: Ground truth class indices
        
        Returns:
            Loss tensor
        """
        # Convert targets to integers if they're not already
        if targets.data.ndim > 1 and targets.data.shape[1] > 1:
            # One-hot encoded targets
            target_indices = np.argmax(targets.data, axis=1)
        else:
            # Class indices
            target_indices = targets.data.astype(np.int32).reshape(-1)
        
        batch_size = logits.data.shape[0]
        num_classes = logits.data.shape[1]
        
        # Calculate errors for each logit
        errors = np.zeros_like(logits.data)
        
        for i in range(batch_size):
            target_idx = target_indices[i]
            target_logit = logits.data[i, target_idx]
            
            for j in range(num_classes):
                if j != target_idx:
                    # Calculate error max(0, yi - yl + μ)
                    errors[i, j] = max(0, logits.data[i, j] - target_logit + self.margin)
        
        # For each sample, threshold errors by mean and compute mean of above-zero values
        sample_losses = []
        for i in range(batch_size):
            # Get non-zero errors
            non_zero_errors = errors[i][errors[i] > 0]
            if len(non_zero_errors) == 0:
                sample_losses.append(0.0)
                continue
                
            # Calculate mean error
            mean_error = np.mean(non_zero_errors)
            
            # Set errors below mean to zero
            above_mean_errors = non_zero_errors[non_zero_errors >= mean_error]
            
            if len(above_mean_errors) > 0:
                # Calculate mean of above-mean errors
                sample_losses.append(np.mean(above_mean_errors))
            else:
                sample_losses.append(0.0)
        
        # Calculate final loss (mean of all sample losses)
        total_loss = np.mean(sample_losses)
        
        # Create output tensor
        out = Tensor(total_loss, requires_grad=True)
        
        def _backward():
            if logits.requires_grad and out.grad is not None:
                # Initialize gradient
                grad = np.zeros_like(logits.data)
                
                for i in range(batch_size):
                    target_idx = target_indices[i]
                    target_logit = logits.data[i, target_idx]
                    
                    # Get errors for this sample
                    sample_errors = errors[i]
                    non_zero_errors = sample_errors[sample_errors > 0]
                    
                    if len(non_zero_errors) == 0:
                        continue
                        
                    # Calculate mean error
                    mean_error = np.mean(non_zero_errors)
                    
                    # Count errors above mean
                    above_mean_mask = sample_errors >= mean_error
                    num_above_mean = np.sum(above_mean_mask)
                    
                    if num_above_mean == 0:
                        continue
                    
                    # Calculate gradients
                    for j in range(num_classes):
                        if j != target_idx and above_mean_mask[j]:
                            # Gradient for non-target logits with errors above mean
                            grad[i, j] += 1.0 / num_above_mean
                            # Gradient for target logit (negative sum of others)
                            grad[i, target_idx] -= 1.0 / num_above_mean
                
                # Apply upstream gradient
                grad = grad * (out.grad / batch_size)
                
                # Accumulate gradient
                logits.grad = grad if logits.grad is None else logits.grad + grad
        
        out._backward = _backward
        out._prev = {logits}
        return out


class HEMPlusLoss(HEMLoss):
    """
    HEM+ loss: HEM with margins adjusted based on class frequency.
    """
    def __init__(self, class_counts=None, base_margin=2000):
        """
        Initialize HEM+ loss.
        
        Args:
            class_counts: Number of samples for each class
            base_margin: Base M value for margin calculation
        """
        super().__init__(margin=0.5)  # Default margin will be overridden
        self.class_counts = class_counts
        self.base_margin = base_margin
        self.margins = None
        
        # Calculate per-class margins if class counts are provided
        if class_counts is not None:
            self._calculate_margins()
    
    def _calculate_margins(self):
        """Calculate margins based on class counts"""
        if self.class_counts is None:
            return
            
        total_samples = sum(self.class_counts)
        n_classes = len(self.class_counts)
        
        # Calculate margins based on formula μi = √(M/(n*si))
        self.margins = np.array([
            np.sqrt(self.base_margin / (n_classes * count)) 
            for count in self.class_counts
        ])
    
    def set_class_counts(self, class_counts):
        """Set class counts and recalculate margins"""
        self.class_counts = class_counts
        self._calculate_margins()
    
    def _multiclass_loss(self, logits: Tensor, targets: Tensor):
        """
        Compute HEM+ loss with class-adjusted margins for multi-class.
        
        Args:
            logits: Raw output from the network (pre-softmax)
            targets: Ground truth class indices
        
        Returns:
            Loss tensor
        """
        # For balanced datasets, fall back to standard HEM
        if self.margins is None:
            return super()._multiclass_loss(logits, targets)
        
        # Convert targets to integers if they're not already
        if targets.data.ndim > 1 and targets.data.shape[1] > 1:
            # One-hot encoded targets
            target_indices = np.argmax(targets.data, axis=1)
        else:
            # Class indices
            target_indices = targets.data.astype(np.int32).reshape(-1)
        
        batch_size = logits.data.shape[0]
        num_classes = logits.data.shape[1]
        
        # Calculate errors for each logit with class-specific margins
        errors = np.zeros_like(logits.data)
        
        for i in range(batch_size):
            target_idx = target_indices[i]
            target_logit = logits.data[i, target_idx]
            
            for j in range(num_classes):
                if j != target_idx:
                    # Use class-specific margin for each non-target logit
                    margin = self.margins[j]
                    errors[i, j] = max(0, logits.data[i, j] - target_logit + margin)
        
        # Rest of the implementation is the same as HEM
        # For each sample, threshold errors by mean and compute mean of above-zero values
        sample_losses = []
        for i in range(batch_size):
            # Get non-zero errors
            non_zero_errors = errors[i][errors[i] > 0]
            if len(non_zero_errors) == 0:
                sample_losses.append(0.0)
                continue
                
            # Calculate mean error
            mean_error = np.mean(non_zero_errors)
            
            # Set errors below mean to zero
            above_mean_errors = non_zero_errors[non_zero_errors >= mean_error]
            
            if len(above_mean_errors) > 0:
                # Calculate mean of above-mean errors
                sample_losses.append(np.mean(above_mean_errors))
            else:
                sample_losses.append(0.0)
        
        # Calculate final loss (mean of all sample losses)
        total_loss = np.mean(sample_losses)
        
        # Create output tensor
        out = Tensor(total_loss, requires_grad=True)
        
        def _backward():
            if logits.requires_grad and out.grad is not None:
                # Initialize gradient
                grad = np.zeros_like(logits.data)
                
                for i in range(batch_size):
                    target_idx = target_indices[i]
                    target_logit = logits.data[i, target_idx]
                    
                    # Get errors for this sample
                    sample_errors = errors[i]
                    non_zero_errors = sample_errors[sample_errors > 0]
                    
                    if len(non_zero_errors) == 0:
                        continue
                        
                    # Calculate mean error
                    mean_error = np.mean(non_zero_errors)
                    
                    # Count errors above mean
                    above_mean_mask = sample_errors >= mean_error
                    num_above_mean = np.sum(above_mean_mask)
                    
                    if num_above_mean == 0:
                        continue
                    
                    # Calculate gradients
                    for j in range(num_classes):
                        if j != target_idx and above_mean_mask[j]:
                            # Gradient for non-target logits with errors above mean
                            grad[i, j] += 1.0 / num_above_mean
                            # Gradient for target logit (negative sum of others)
                            grad[i, target_idx] -= 1.0 / num_above_mean
                
                # Apply upstream gradient
                grad = grad * (out.grad / batch_size)
                
                # Accumulate gradient
                logits.grad = grad if logits.grad is None else logits.grad + grad
        
        out._backward = _backward
        out._prev = {logits}
        return out
        
    def _binary_loss(self, logits: Tensor, targets: Tensor):
        """
        HEM+ loss for binary classification with class-adjusted margins
        """
        # For balanced datasets, fall back to standard HEM
        if self.margins is None:
            return super()._binary_loss(logits, targets)
            
        # Otherwise, implement the adjusted margin version for binary
        # (Implementation follows similar pattern to the multiclass version)
        
        batch_size = logits.data.shape[0]
        
        # Convert targets to flat array if needed
        if targets.data.ndim > 1:
            target_values = targets.data.flatten()
        else:
            target_values = targets.data
            
        # Compute errors for each sample
        errors = np.zeros((batch_size, 2))  # Two errors: one for each class
        
        for i in range(batch_size):
            pred = logits.data[i, 0]
            target = target_values[i]
            
            # If target is 1, the prediction should be high (positive)
            # If target is 0, the prediction should be low (negative)
            margin_0 = self.margins[0] if self.margins is not None else self.margin
            margin_1 = self.margins[1] if self.margins is not None else self.margin
            
            if target > 0.5:  # Target is 1
                # Error if prediction is too low
                errors[i, 0] = max(0, margin_1 - pred)
            else:  # Target is 0
                # Error if prediction is too high
                errors[i, 1] = max(0, pred + margin_0)
                
        # For each sample, threshold errors by mean and compute mean of above-zero values
        sample_losses = []
        for i in range(batch_size):
            # Get non-zero errors
            non_zero_errors = errors[i][errors[i] > 0]
            if len(non_zero_errors) == 0:
                sample_losses.append(0.0)
                continue
                
            # Calculate mean error
            mean_error = np.mean(non_zero_errors)
            
            # Set errors below mean to zero
            above_mean_errors = non_zero_errors[non_zero_errors >= mean_error]
            
            if len(above_mean_errors) > 0:
                # Calculate mean of above-mean errors
                sample_losses.append(np.mean(above_mean_errors))
            else:
                sample_losses.append(0.0)
        
        # Calculate final loss (mean of all sample losses)
        total_loss = np.mean(sample_losses)
        
        # Create output tensor
        out = Tensor(total_loss, requires_grad=True)
        
        def _backward():
            if logits.requires_grad and out.grad is not None:
                # Initialize gradient
                grad = np.zeros_like(logits.data)
                
                for i in range(batch_size):
                    pred = logits.data[i, 0]
                    target = target_values[i]
                    
                    # Calculate sample errors
                    sample_errors = errors[i]
                    non_zero_errors = sample_errors[sample_errors > 0]
                    
                    if len(non_zero_errors) == 0:
                        continue
                        
                    # Calculate mean error
                    mean_error = np.mean(non_zero_errors)
                    
                    # Count errors above mean
                    above_mean_mask = sample_errors >= mean_error
                    num_above_mean = np.sum(above_mean_mask)
                    
                    if num_above_mean == 0:
                        continue
                    
                    # Calculate gradient based on which error(s) are above mean
                    if target > 0.5:  # Target is 1
                        if above_mean_mask[0]:  # Error for target 1 is above mean
                            grad[i, 0] -= 1.0 / num_above_mean  # Increase prediction
                    else:  # Target is 0
                        if above_mean_mask[1]:  # Error for target 0 is above mean
                            grad[i, 0] += 1.0 / num_above_mean  # Decrease prediction
                
                # Apply upstream gradient
                grad = grad * (out.grad / batch_size)
                
                # Accumulate gradient
                logits.grad = grad if logits.grad is None else logits.grad + grad
        
        out._backward = _backward
        out._prev = {logits}
        return out