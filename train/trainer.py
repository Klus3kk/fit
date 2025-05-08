"""
This module implements the Trainer class for training and evaluating machine learning models.
"""

import time

import numpy as np

import utils.regularization as reg
from core.tensor import Tensor
from monitor.tracker import TrainingTracker


class Trainer:
    """
    Trainer class that handles model training, evaluation, and monitoring.

    Provides utilities for training with mini-batches, regularization,
    gradient clipping, and tracking metrics during training.
    """

    def __init__(
        self, model, loss_fn, optimizer, tracker=None, scheduler=None, grad_clip=None
    ):
        """
        Initialize a Trainer instance.

        Args:
            model: Model to train
            loss_fn: Loss function
            optimizer: Optimizer for parameter updates
            tracker: Training tracker for monitoring metrics (optional)
            scheduler: Learning rate scheduler (optional)
            grad_clip: Maximum gradient norm for clipping (optional)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.tracker = tracker
        self.scheduler = scheduler
        self.grad_clip = grad_clip  # Maximum gradient norm

    def _set_training_mode(self, training=True):
        """
        Set all modules to training or evaluation mode.

        Args:
            training: If True, set to training mode; otherwise, set to evaluation mode
        """

        def set_mode(module):
            if hasattr(module, "training"):
                module.training = training
            if hasattr(module, "train") and training:
                module.train()
            if hasattr(module, "eval") and not training:
                module.eval()
            if hasattr(module, "_children"):
                for child in module._children:
                    set_mode(child)

        set_mode(self.model)

    def _clip_gradients(self):
        """
        Apply gradient clipping to all model parameters.

        Prevents exploding gradients by limiting the gradient norm
        to the value specified by self.grad_clip.
        """
        if self.grad_clip is None:
            return

        for param in self.model.parameters():
            if param.grad is not None:
                param.clip_gradients(self.grad_clip)

    def fit(self, x, y, epochs=10, batch_size=None, verbose=True, l2_lambda=0):
        """
        Train the model with optional batching and regularization.

        Args:
            x: Input tensor
            y: Target tensor
            epochs: Number of training epochs
            batch_size: Size of mini-batches (None for full batch training)
            verbose: Whether to print progress
            l2_lambda: L2 regularization strength (0 for no regularization)

        Returns:
            None - training is performed in-place on the model
        """
        n_samples = len(x.data)

        # Enable training mode
        self._set_training_mode(True)

        for epoch in range(1, epochs + 1):
            if self.tracker:
                self.tracker.start_epoch()

            # If batch size is set, use mini-batch training
            if batch_size is None:
                # Full batch training
                preds = self.model(x)
                loss = self.loss_fn(preds, y)

                # Add L2 regularization if specified
                if l2_lambda > 0:
                    loss = reg.apply_l2_regularization(self.model, loss, l2_lambda)

                loss.backward()

                # Apply gradient clipping if specified
                self._clip_gradients()

                self.optimizer.step()
                self.optimizer.zero_grad()

                # Calculate accuracy for classification tasks
                acc = self._calculate_accuracy(preds, y)

            else:
                # Mini-batch training
                total_loss = 0
                correct = 0
                total = 0

                # Shuffle indices for each epoch
                indices = np.random.permutation(n_samples)

                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    batch_indices = indices[start_idx:end_idx]

                    # Extract batch data
                    x_batch = Tensor(x.data[batch_indices], requires_grad=True)
                    y_batch = Tensor(y.data[batch_indices], requires_grad=False)

                    # Forward pass
                    preds_batch = self.model(x_batch)
                    loss_batch = self.loss_fn(preds_batch, y_batch)

                    # Add L2 regularization if specified
                    if l2_lambda > 0:
                        loss_batch = reg.apply_l2_regularization(
                            self.model, loss_batch, l2_lambda
                        )

                    # Backward pass
                    loss_batch.backward()

                    # Apply gradient clipping if specified
                    self._clip_gradients()

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Accumulate loss and accuracy
                    total_loss += loss_batch.data * (end_idx - start_idx)

                    # Calculate batch accuracy for classification tasks
                    if preds_batch.data.ndim == 2 and preds_batch.data.shape[1] > 1:
                        pred_labels = preds_batch.data.argmax(axis=1)
                        correct += (pred_labels == y_batch.data).sum()
                        total += len(y_batch.data)

                # Calculate average loss and accuracy
                avg_loss = total_loss / n_samples
                acc = correct / total if total > 0 else None
                loss = Tensor(avg_loss)  # For logging

            # Step the learning rate scheduler if provided
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_lr()
            else:
                current_lr = self.optimizer.lr

            # Log metrics
            if self.tracker:
                self.tracker.log(loss=loss.data, acc=acc, lr=current_lr)

            # Print progress
            if verbose:
                acc_str = f"{acc * 100:.2f}%" if acc is not None else "-"
                print("╭" + "─" * 50 + "╮")
                print(
                    "| "
                    + f"Epoch {epoch:03d} | Loss: {loss.data:.4f} | Acc: {acc_str:>6} | LR: {current_lr:.4f}"
                    + " |"
                )
                print("╰" + "─" * 50 + "╯")

        # Print training summary
        if self.tracker:
            self.tracker.summary()

    def _calculate_accuracy(self, predictions, targets):
        """
        Calculate accuracy for classification tasks.

        Args:
            predictions: Model predictions
            targets: Ground truth labels

        Returns:
            Accuracy as a float between 0 and 1, or None if not applicable
        """
        if predictions.data.ndim == 2 and predictions.data.shape[1] > 1:
            # Multi-class classification
            predicted_labels = predictions.data.argmax(axis=1)
            correct = (predicted_labels == targets.data).sum()
            return correct / len(targets.data)
        return None

    def evaluate(self, x, y, batch_size=None):
        """
        Evaluate the model on test data.

        Args:
            x: Input tensor
            y: Target tensor
            batch_size: Size of mini-batches (None for full batch evaluation)

        Returns:
            Tuple of (loss, accuracy)
        """
        # Set to evaluation mode
        self._set_training_mode(False)

        n_samples = len(x.data)

        if batch_size is None:
            # Full batch evaluation
            preds = self.model(x)
            loss = self.loss_fn(preds, y)
            acc = self._calculate_accuracy(preds, y)

            return float(loss.data), acc
        else:
            # Mini-batch evaluation
            total_loss = 0
            correct = 0
            total = 0

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)

                # Extract batch data
                x_batch = Tensor(x.data[start_idx:end_idx], requires_grad=False)
                y_batch = Tensor(y.data[start_idx:end_idx], requires_grad=False)

                # Forward pass
                preds_batch = self.model(x_batch)
                loss_batch = self.loss_fn(preds_batch, y_batch)

                # Accumulate loss and accuracy
                total_loss += loss_batch.data * (end_idx - start_idx)

                # Calculate batch accuracy for classification tasks
                if preds_batch.data.ndim == 2 and preds_batch.data.shape[1] > 1:
                    pred_labels = preds_batch.data.argmax(axis=1)
                    correct += (pred_labels == y_batch.data).sum()
                    total += len(y_batch.data)

            # Calculate average loss and accuracy
            avg_loss = total_loss / n_samples
            acc = correct / total if total > 0 else None

            return float(avg_loss), acc
