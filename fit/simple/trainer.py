"""
High-level training API for FIT framework.

This module provides simple, one-line training functions that handle
all the boilerplate while remaining flexible for advanced use cases.
"""

import numpy as np
from typing import Union, Optional, List, Dict, Any, Callable
import time

from core.tensor import Tensor
from nn.modules.container import Sequential
from nn.modules.linear import Linear
from nn.modules.activation import ReLU, Softmax, Tanh
from nn.modules.normalization import BatchNorm
from loss.classification import CrossEntropyLoss
from loss.regression import MSELoss
from optim.adam import Adam
from optim.sgd import SGD, SGDMomentum
from optim.experimental.sam import SAM
from data.dataset import Dataset
from data.dataloader import DataLoader
from monitor.tracker import TrainingTracker


class SimpleTrainer:
    """
    High-level trainer that handles all the boilerplate.

    Makes training as simple as:
    trainer = SimpleTrainer(model, data)
    trainer.fit()
    """

    def __init__(
        self,
        model,
        data: Union[tuple, DataLoader],
        validation_data: Optional[Union[tuple, DataLoader]] = None,
        loss: Union[str, Any] = "auto",
        optimizer: Union[str, Any] = "adam",
        metrics: Optional[List[str]] = None,
        callbacks: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the simple trainer.

        Args:
            model: The model to train
            data: Training data as (X, y) tuple or DataLoader
            validation_data: Validation data (optional)
            loss: Loss function ('auto', 'mse', 'crossentropy', or loss object)
            optimizer: Optimizer ('adam', 'sgd', 'sam', or optimizer object)
            metrics: List of metrics to track ['accuracy', 'loss']
            callbacks: List of callback names ['early_stopping', 'lr_scheduler']
            **kwargs: Additional arguments (lr, batch_size, etc.)
        """
        self.model = model
        self.kwargs = kwargs

        # Set up data
        self.train_loader = self._setup_data(data, shuffle=True)
        self.val_loader = (
            self._setup_data(validation_data, shuffle=False)
            if validation_data
            else None
        )

        # Set up loss function
        self.loss_fn = self._setup_loss(loss)

        # Set up optimizer
        self.optimizer = self._setup_optimizer(optimizer)

        # Set up metrics
        self.metrics = metrics or ["loss", "accuracy"]

        # Set up callbacks
        self.callbacks = self._setup_callbacks(callbacks or [])

        # Set up tracker
        self.tracker = TrainingTracker(
            experiment_name=kwargs.get("experiment_name"),
            early_stopping=self._get_early_stopping_config(),
        )

    def _setup_data(self, data, shuffle=True):
        """Convert data to DataLoader if needed."""
        if data is None:
            return None

        if isinstance(data, DataLoader):
            return data

        if isinstance(data, tuple) and len(data) == 2:
            X, y = data
            batch_size = self.kwargs.get("batch_size", 32)
            dataset = Dataset(X, y)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        raise ValueError("Data must be (X, y) tuple or DataLoader")

    def _setup_loss(self, loss):
        """Set up loss function with smart defaults."""
        if isinstance(loss, str):
            if loss == "auto":
                # Auto-detect based on model output
                return self._auto_detect_loss()
            elif loss == "mse":
                return MSELoss()
            elif loss == "crossentropy":
                return CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown loss: {loss}")
        else:
            return loss

    def _auto_detect_loss(self):
        """Automatically detect appropriate loss function."""
        # Simple heuristic: if output size > 1, assume classification
        # This could be made smarter by examining the data
        try:
            # Get a sample to test model output
            sample_x, sample_y = next(iter(self.train_loader))
            output = self.model(sample_x[:1])  # Single sample

            if output.data.shape[-1] > 1:
                return CrossEntropyLoss()
            else:
                return MSELoss()
        except:
            # Default to MSE if we can't determine
            return MSELoss()

    def _setup_optimizer(self, optimizer):
        """Set up optimizer with smart defaults."""
        lr = self.kwargs.get("lr", 0.001)

        if isinstance(optimizer, str):
            if optimizer == "adam":
                return Adam(self.model.parameters(), lr=lr)
            elif optimizer == "sgd":
                momentum = self.kwargs.get("momentum", 0.9)
                return SGDMomentum(self.model.parameters(), lr=lr, momentum=momentum)
            elif optimizer == "sam":
                base_opt = Adam(self.model.parameters(), lr=lr)
                rho = self.kwargs.get("rho", 0.05)
                return SAM(self.model.parameters(), base_opt, rho=rho)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            return optimizer

    def _setup_callbacks(self, callback_names):
        """Set up callbacks from names."""
        callbacks = {}

        for name in callback_names:
            if name == "early_stopping":
                callbacks["early_stopping"] = {
                    "patience": self.kwargs.get("patience", 10),
                    "min_delta": self.kwargs.get("min_delta", 0.001),
                }
            elif name == "lr_scheduler":
                callbacks["lr_scheduler"] = {
                    "factor": self.kwargs.get("lr_factor", 0.5),
                    "patience": self.kwargs.get("lr_patience", 5),
                }

        return callbacks

    def _get_early_stopping_config(self):
        """Get early stopping configuration."""
        if "early_stopping" in self.callbacks:
            return self.callbacks["early_stopping"]
        return None

    def fit(
        self,
        epochs: int = 10,
        verbose: bool = True,
        validation_freq: int = 1,
        save_best: bool = True,
        **kwargs,
    ):
        """
        Train the model.

        Args:
            epochs: Number of epochs to train
            verbose: Whether to print progress
            validation_freq: How often to run validation
            save_best: Whether to save the best model
            **kwargs: Additional arguments
        """
        if verbose:
            print(f"🚀 Starting training for {epochs} epochs...")
            print(f"📊 Model: {self.model.__class__.__name__}")
            print(f"🔧 Optimizer: {self.optimizer.__class__.__name__}")
            print(f"📉 Loss: {self.loss_fn.__class__.__name__}")
            if self.val_loader:
                print(f"✅ Validation data: {len(self.val_loader)} batches")
            print("-" * 50)

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(1, epochs + 1):
            # Start epoch timing
            self.tracker.start_epoch()

            # Training phase
            train_metrics = self._train_epoch()

            # Validation phase
            val_metrics = {}
            if self.val_loader and epoch % validation_freq == 0:
                val_metrics = self._validate_epoch()

                # Save best model
                if save_best and val_metrics.get("loss", float("inf")) < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_model_state = self._get_model_state()

            # Update learning rate if scheduler is enabled
            if "lr_scheduler" in self.callbacks:
                self._update_lr_scheduler(val_metrics.get("loss"))

            # Log metrics
            current_lr = getattr(self.optimizer, "lr", None)
            custom_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

            self.tracker.log(
                loss=train_metrics["loss"],
                acc=train_metrics.get("accuracy"),
                lr=current_lr,
                custom_metrics=custom_metrics,
            )

            # Print progress
            if verbose:
                self._print_epoch_progress(epoch, epochs, train_metrics, val_metrics)

            # Check early stopping
            if self.tracker.should_early_stop():
                if verbose:
                    print(f"\n⏰ Early stopping triggered at epoch {epoch}")
                break

        # Restore best model if saved
        if save_best and best_model_state:
            self._restore_model_state(best_model_state)
            if verbose:
                print(f"📦 Restored best model (val_loss: {best_val_loss:.4f})")

        if verbose:
            print("\n🎉 Training completed!")
            self.tracker.summary(style="box")

        return self.tracker

    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for x, y in self.train_loader:
            # Handle SAM optimizer
            if isinstance(self.optimizer, SAM):

                def closure():
                    self.optimizer.zero_grad()
                    outputs = self.model(x)
                    loss = self.loss_fn(outputs, y)
                    loss.backward()
                    return loss

                loss = self.optimizer.first_step(closure)
                loss = self.optimizer.second_step(closure)
                outputs = self.model(x)
            else:
                # Standard training
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                loss.backward()
                self.optimizer.step()

            # Update metrics
            batch_size = x.data.shape[0]
            total_loss += float(loss.data) * batch_size
            total += batch_size

            # Calculate accuracy for classification
            if (
                "accuracy" in self.metrics
                and outputs.data.ndim > 1
                and outputs.data.shape[1] > 1
            ):
                predictions = np.argmax(outputs.data, axis=1)
                targets = (
                    y.data.astype(np.int32)
                    if y.data.ndim == 1
                    else np.argmax(y.data, axis=1)
                )
                correct += np.sum(predictions == targets)

        metrics = {
            "loss": total_loss / total,
        }

        if "accuracy" in self.metrics and total > 0:
            metrics["accuracy"] = correct / total

        return metrics

    def _validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        for x, y in self.val_loader:
            outputs = self.model(x)
            loss = self.loss_fn(outputs, y)

            # Update metrics
            batch_size = x.data.shape[0]
            total_loss += float(loss.data) * batch_size
            total += batch_size

            # Calculate accuracy for classification
            if (
                "accuracy" in self.metrics
                and outputs.data.ndim > 1
                and outputs.data.shape[1] > 1
            ):
                predictions = np.argmax(outputs.data, axis=1)
                targets = (
                    y.data.astype(np.int32)
                    if y.data.ndim == 1
                    else np.argmax(y.data, axis=1)
                )
                correct += np.sum(predictions == targets)

        metrics = {
            "loss": total_loss / total,
        }

        if "accuracy" in self.metrics and total > 0:
            metrics["accuracy"] = correct / total

        return metrics

    def _print_epoch_progress(self, epoch, total_epochs, train_metrics, val_metrics):
        """Print training progress."""
        progress = f"Epoch {epoch:3d}/{total_epochs}"

        # Training metrics
        train_str = f"train_loss: {train_metrics['loss']:.4f}"
        if "accuracy" in train_metrics:
            train_str += f", train_acc: {train_metrics['accuracy']:.3f}"

        # Validation metrics
        val_str = ""
        if val_metrics:
            val_str = f", val_loss: {val_metrics['loss']:.4f}"
            if "accuracy" in val_metrics:
                val_str += f", val_acc: {val_metrics['accuracy']:.3f}"

        print(f"{progress} - {train_str}{val_str}")

    def _update_lr_scheduler(self, val_loss):
        """Update learning rate based on validation loss."""
        # Simple implementation - reduce LR on plateau
        # This could be made more sophisticated
        pass

    def _get_model_state(self):
        """Get current model state for saving."""
        # Simple state saving - could be improved
        return {param: param.data.copy() for param in self.model.parameters()}

    def _restore_model_state(self, state):
        """Restore model state."""
        for param, saved_data in zip(self.model.parameters(), state.values()):
            param.data = saved_data


def train(
    model,
    data: Union[tuple, DataLoader],
    epochs: int = 10,
    validation_data: Optional[Union[tuple, DataLoader]] = None,
    loss: str = "auto",
    optimizer: str = "adam",
    lr: float = 0.001,
    batch_size: int = 32,
    callbacks: Optional[List[str]] = None,
    verbose: bool = True,
    **kwargs,
) -> TrainingTracker:
    """
    High-level training function - train any model in one line!

    Args:
        model: Model to train
        data: Training data as (X, y) tuple or DataLoader
        epochs: Number of epochs
        validation_data: Validation data (optional)
        loss: Loss function ('auto', 'mse', 'crossentropy')
        optimizer: Optimizer ('adam', 'sgd', 'sam')
        lr: Learning rate
        batch_size: Batch size (if data is tuple)
        callbacks: List of callbacks ['early_stopping', 'lr_scheduler']
        verbose: Whether to print progress
        **kwargs: Additional arguments

    Returns:
        TrainingTracker with training history

    Examples:
        # Simple training
        >>> tracker = fit.train(model, (X_train, y_train), epochs=50)

        # With validation and callbacks
        >>> tracker = fit.train(
        ...     model, (X_train, y_train),
        ...     validation_data=(X_val, y_val),
        ...     optimizer='sam',
        ...     callbacks=['early_stopping'],
        ...     epochs=100
        ... )
    """
    trainer = SimpleTrainer(
        model=model,
        data=data,
        validation_data=validation_data,
        loss=loss,
        optimizer=optimizer,
        lr=lr,
        batch_size=batch_size,
        callbacks=callbacks,
        **kwargs,
    )

    return trainer.fit(epochs=epochs, verbose=verbose)


def quick_evaluate(
    model, data: Union[tuple, DataLoader], metrics: List[str] = None
) -> Dict[str, float]:
    """
    Quickly evaluate a model on test data.

    Args:
        model: Trained model
        data: Test data as (X, y) tuple or DataLoader
        metrics: Metrics to compute ['loss', 'accuracy']

    Returns:
        Dictionary of computed metrics
    """
    if isinstance(data, tuple):
        X, y = data
        dataset = Dataset(X, y)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    else:
        data_loader = data

    model.eval()
    metrics = metrics or ["loss", "accuracy"]

    # Auto-detect loss function
    sample_x, sample_y = next(iter(data_loader))
    output = model(sample_x[:1])
    if output.data.shape[-1] > 1:
        loss_fn = CrossEntropyLoss()
    else:
        loss_fn = MSELoss()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in data_loader:
        outputs = model(x)

        if "loss" in metrics:
            loss = loss_fn(outputs, y)
            total_loss += float(loss.data) * x.data.shape[0]

        if "accuracy" in metrics and outputs.data.ndim > 1:
            predictions = np.argmax(outputs.data, axis=1)
            targets = (
                y.data.astype(np.int32)
                if y.data.ndim == 1
                else np.argmax(y.data, axis=1)
            )
            correct += np.sum(predictions == targets)

        total += x.data.shape[0]

    results = {}
    if "loss" in metrics:
        results["loss"] = total_loss / total
    if "accuracy" in metrics and total > 0:
        results["accuracy"] = correct / total

    return results
