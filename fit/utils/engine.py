# Example implementation for train/engine.py
from typing import Any, Callable, Dict, Optional

from fit.monitor.tracker import TrainingTracker
from fit.simple.trainer import Trainer
from fit.data import DataLoader


def train_epoch(model, dataloader, loss_fn, optimizer, device=None):
    """
    Train for a single epoch.

    Args:
        model: Model to train
        dataloader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to use (not used in this version, but kept for PyTorch compatibility)

    Returns:
        Dict with epoch metrics (loss, accuracy)
    """
    # Set model to training mode (affects Dropout, BatchNorm, etc.)
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    # Iterate over batches
    for x, y in dataloader:
        # Forward pass
        outputs = model(x)
        loss = loss_fn(outputs, y)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update metrics
        total_loss += loss.data

        # Calculate accuracy for classification tasks
        if outputs.data.ndim > 1 and outputs.data.shape[1] > 1:
            # Multi-class classification
            predictions = outputs.data.argmax(axis=1)
            correct += (predictions == y.data).sum()
            total += len(y.data)

    # Calculate epoch metrics
    metrics = {
        "loss": total_loss / len(dataloader),
    }

    if total > 0:
        metrics["accuracy"] = correct / total

    return metrics


def evaluate(model, dataloader, loss_fn, device=None):
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        loss_fn: Loss function
        device: Device to use (not used in this version, but kept for PyTorch compatibility)

    Returns:
        Dict with evaluation metrics (loss, accuracy)
    """
    # Set model to evaluation mode (affects Dropout, BatchNorm, etc.)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Iterate over batches (no gradient tracking needed)
    for x, y in dataloader:
        # Forward pass
        outputs = model(x)
        loss = loss_fn(outputs, y)

        # Update metrics
        total_loss += loss.data

        # Calculate accuracy for classification tasks
        if outputs.data.ndim > 1 and outputs.data.shape[1] > 1:
            # Multi-class classification
            predictions = outputs.data.argmax(axis=1)
            correct += (predictions == y.data).sum()
            total += len(y.data)

    # Calculate evaluation metrics
    metrics = {
        "loss": total_loss / len(dataloader),
    }

    if total > 0:
        metrics["accuracy"] = correct / total

    return metrics


def train(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    epochs=10,
    device=None,
    scheduler=None,
    early_stopping=None,
    tracker=None,
):
    """
    Complete training loop.

    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        loss_fn: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        device: Device to use (not used in this version, but kept for PyTorch compatibility)
        scheduler: Learning rate scheduler (optional)
        early_stopping: Early stopping settings (optional)
        tracker: TrainingTracker for logging (optional)

    Returns:
        TrainingTracker with training history
    """
    # Create tracker if none provided
    if tracker is None:
        tracker = TrainingTracker(early_stopping=early_stopping)

    for epoch in range(1, epochs + 1):
        # Start epoch
        tracker.start_epoch()

        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)

        # Evaluate on validation set
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, loss_fn, device)

        # Update learning rate if scheduler provided
        if scheduler is not None:
            scheduler.step()

        # Log metrics
        custom_metrics = {}
        if val_metrics:
            custom_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

        tracker.log(
            loss=train_metrics["loss"],
            acc=train_metrics.get("accuracy"),
            lr=optimizer.lr if hasattr(optimizer, "lr") else None,
            custom_metrics=custom_metrics,
        )

        # Print progress
        tracker.summary(last_n=1, style="box")

        # Check early stopping
        if tracker.should_early_stop():
            print("Early stopping triggered!")
            break

    return tracker
