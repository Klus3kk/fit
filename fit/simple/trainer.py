"""
High-level training API for FIT framework.

This module provides simple, one-line training functions that handle
all the boilerplate while remaining flexible for advanced use cases.
"""

import numpy as np
from typing import Union, Optional, List, Dict, Any, Callable
import time

from fit.core.tensor import Tensor
from fit.nn.modules.container import Sequential
from fit.nn.modules.linear import Linear
from fit.nn.modules.activation import ReLU, Softmax, Tanh
from fit.nn.modules.normalization import BatchNorm
from fit.loss.classification import CrossEntropyLoss
from fit.loss.regression import MSELoss
from fit.optim.adam import Adam
from fit.optim.sgd import SGD, SGDMomentum
from fit.optim.experimental.sam import SAM
from fit.data.dataset import Dataset
from fit.data.dataloader import DataLoader
from fit.monitor.tracker import TrainingTracker


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
            dataset = Dataset(X, y)
            batch_size = self.kwargs.get("batch_size", 32)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        raise ValueError("Data must be a tuple (X, y) or DataLoader")

    def _setup_loss(self, loss):
        """Set up loss function."""
        if isinstance(loss, str):
            if loss == "auto":
                # Try to infer from model output size
                return self._infer_loss()
            elif loss.lower() in ["mse", "mean_squared_error"]:
                return MSELoss()
            elif loss.lower() in ["crossentropy", "cross_entropy", "ce"]:
                return CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown loss function: {loss}")
        else:
            return loss

    def _infer_loss(self):
        """Infer appropriate loss function from model."""
        # Get a sample from the data to test model output
        try:
            sample_batch = next(iter(self.train_loader))
            sample_x, sample_y = sample_batch

            output = self.model(sample_x)
            output_size = output.data.shape[-1]

            # If output is single value, use MSE
            if output_size == 1:
                return MSELoss()
            # If output is multi-class, use CrossEntropy
            else:
                return CrossEntropyLoss()

        except Exception:
            # Default to MSE if we can't infer
            print("Warning: Could not infer loss function, defaulting to MSE")
            return MSELoss()

    def _setup_optimizer(self, optimizer):
        """Set up optimizer."""
        lr = self.kwargs.get("lr", 0.001)

        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                return Adam(self.model.parameters(), lr=lr)
            elif optimizer.lower() == "sgd":
                momentum = self.kwargs.get("momentum", 0.0)
                if momentum > 0:
                    return SGDMomentum(
                        self.model.parameters(), lr=lr, momentum=momentum
                    )
                else:
                    return SGD(self.model.parameters(), lr=lr)
            elif optimizer.lower() == "sam":
                base_opt = self.kwargs.get("base_optimizer", "sgd")
                rho = self.kwargs.get("rho", 0.05)

                if base_opt.lower() == "adam":
                    base = Adam(self.model.parameters(), lr=lr)
                else:
                    base = SGD(self.model.parameters(), lr=lr)

                return SAM(self.model.parameters(), base, rho=rho)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            return optimizer

    def _setup_callbacks(self, callbacks):
        """Set up training callbacks."""
        callback_instances = []

        for callback in callbacks:
            if callback == "early_stopping":
                # Early stopping will be handled by tracker
                continue
            elif callback == "lr_scheduler":
                # Could implement learning rate scheduling here
                continue

        return callback_instances

    def _get_early_stopping_config(self):
        """Get early stopping configuration."""
        if "early_stopping" in self.kwargs:
            return {
                "patience": self.kwargs.get("patience", 10),
                "min_delta": self.kwargs.get("min_delta", 1e-4),
                "metric": self.kwargs.get("monitor", "val_loss"),
            }
        return None

    def fit(self, epochs: int = 100, verbose: int = 1):
        """
        Train the model.

        Args:
            epochs: Number of epochs to train
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)

        Returns:
            Training history dictionary
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Loss: {self.loss_fn.__class__.__name__}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print("-" * 50)

        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(epoch, verbose)
            history["train_loss"].append(train_loss)

            # Validation phase
            if self.val_loader:
                val_loss, val_acc = self._validate_epoch(epoch, verbose)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
            else:
                val_loss, val_acc = None, None

            # Update tracker
            metrics = {"train_loss": train_loss}
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            if val_acc is not None:
                metrics["val_accuracy"] = val_acc

            should_stop = self.tracker.update(epoch, metrics)

            # Print progress
            if verbose >= 1:
                self._print_epoch_results(epoch, train_loss, val_loss, val_acc)

            # Early stopping
            if should_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print("Training completed!")
        return history

    def _train_epoch(self, epoch, verbose):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            # Zero gradients
            for param in self.model.parameters():
                param.grad = None

            # Forward pass
            output = self.model(batch_x)
            loss = self.loss_fn(output, batch_y)

            # Backward pass
            loss.backward()

            # Optimizer step (handle SAM specially)
            if isinstance(self.optimizer, SAM):
                self.optimizer.first_step(zero_grad=True)

                # Second forward pass for SAM
                output2 = self.model(batch_x)
                loss2 = self.loss_fn(output2, batch_y)
                loss2.backward()

                self.optimizer.second_step(zero_grad=True)
            else:
                self.optimizer.step()

            total_loss += loss.data
            batch_count += 1

        return total_loss / batch_count

    def _validate_epoch(self, epoch, verbose):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        for batch_x, batch_y in self.val_loader:
            # Forward pass (no gradients needed)
            output = self.model(batch_x)
            loss = self.loss_fn(output, batch_y)

            total_loss += loss.data
            batch_count += 1

            # Calculate accuracy
            if "accuracy" in self.metrics:
                predictions = np.argmax(output.data, axis=1)
                targets = batch_y.data if hasattr(batch_y, "data") else batch_y

                correct += np.sum(predictions == targets)
                total += len(targets)

        avg_loss = total_loss / batch_count
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _print_epoch_results(self, epoch, train_loss, val_loss, val_acc):
        """Print results for one epoch."""
        msg = f"Epoch {epoch + 1}: train_loss={train_loss:.4f}"

        if val_loss is not None:
            msg += f", val_loss={val_loss:.4f}"
        if val_acc is not None:
            msg += f", val_acc={val_acc:.4f}"

        print(msg)

    def evaluate(self, test_data: Union[tuple, DataLoader]) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_data: Test data as (X, y) tuple or DataLoader

        Returns:
            Dictionary with evaluation metrics
        """
        test_loader = self._setup_data(test_data, shuffle=False)

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        print("Evaluating model...")

        for batch_x, batch_y in test_loader:
            output = self.model(batch_x)
            loss = self.loss_fn(output, batch_y)

            total_loss += loss.data
            batch_count += 1

            # Calculate accuracy
            predictions = np.argmax(output.data, axis=1)
            targets = batch_y.data if hasattr(batch_y, "data") else batch_y

            correct += np.sum(predictions == targets)
            total += len(targets)

        results = {
            "test_loss": total_loss / batch_count,
            "test_accuracy": correct / total,
        }

        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")

        return results

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Input data

        Returns:
            Predictions as numpy array
        """
        self.model.eval()

        if isinstance(X, np.ndarray):
            X = Tensor(X, requires_grad=False)

        output = self.model(X)
        return output.data

    def save(self, filepath: str):
        """Save the trained model."""
        from fit.nn.utils.model_io import save_model

        save_model(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load a trained model."""
        from fit.nn.utils.model_io import load_model

        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")


# Convenience functions for quick training
def fit_classifier(
    model,
    train_data,
    validation_data=None,
    epochs=100,
    lr=0.001,
    batch_size=32,
    optimizer="adam",
    **kwargs,
):
    """
    Quick function to train a classification model.

    Args:
        model: Model to train
        train_data: Training data as (X, y) tuple
        validation_data: Validation data (optional)
        epochs: Number of epochs
        lr: Learning rate
        batch_size: Batch size
        optimizer: Optimizer name or instance
        **kwargs: Additional arguments

    Returns:
        Trained model and training history
    """
    trainer = SimpleTrainer(
        model=model,
        data=train_data,
        validation_data=validation_data,
        loss="crossentropy",
        optimizer=optimizer,
        lr=lr,
        batch_size=batch_size,
        **kwargs,
    )

    history = trainer.fit(epochs=epochs)
    return trainer.model, history


def fit_regressor(
    model,
    train_data,
    validation_data=None,
    epochs=100,
    lr=0.001,
    batch_size=32,
    optimizer="adam",
    **kwargs,
):
    """
    Quick function to train a regression model.

    Args:
        model: Model to train
        train_data: Training data as (X, y) tuple
        validation_data: Validation data (optional)
        epochs: Number of epochs
        lr: Learning rate
        batch_size: Batch size
        optimizer: Optimizer name or instance
        **kwargs: Additional arguments

    Returns:
        Trained model and training history
    """
    trainer = SimpleTrainer(
        model=model,
        data=train_data,
        validation_data=validation_data,
        loss="mse",
        optimizer=optimizer,
        lr=lr,
        batch_size=batch_size,
        **kwargs,
    )

    history = trainer.fit(epochs=epochs)
    return trainer.model, history


def quick_train(model, X, y, **kwargs):
    """
    Ultra-simple training function.

    Args:
        model: Model to train
        X: Input features
        y: Target labels
        **kwargs: Training parameters

    Returns:
        Trained model
    """
    trainer = SimpleTrainer(model=model, data=(X, y), **kwargs)
    trainer.fit(epochs=kwargs.get("epochs", 50))
    return trainer.model
