"""
MNIST digit classification example using the FIT library.

This example demonstrates how to load and train on the MNIST dataset
using the Flexible and Interpretable Training (FIT) library.
"""

import numpy as np
import os
from sklearn.datasets import fetch_openml

from core.tensor import Tensor
from monitor.tracker import TrainingTracker
from nn.activations import Dropout, ReLU, Softmax
from nn.linear import Linear
from nn.model_io import load_model, save_model
from nn.normalization import BatchNorm
from nn.sequential import Sequential
from train.engine import evaluate, train
from train.loss import CrossEntropyLoss
from train.optim import Adam
from train.scheduler import StepLR
from utils.data import DataLoader, Dataset


def load_mnist_data(path="./data/mnist"):
    """
    Load the MNIST dataset using scikit-learn's fetch_openml.

    Args:
        path: Directory to cache the MNIST data

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    # Create data directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    print("Fetching MNIST data from OpenML (this may take a moment)...")
    mnist = fetch_openml(
        "mnist_784", version=1, parser="auto", cache=True, data_home=path
    )

    # Get features and targets
    # Convert from DataFrame/Series to numpy arrays if needed
    if hasattr(mnist.data, "values"):
        X = mnist.data.values.astype("float32") / 255.0
    else:
        X = mnist.data.astype("float32") / 255.0

    if hasattr(mnist.target, "values"):
        y = mnist.target.values.astype("int")
    else:
        y = mnist.target.astype("int")

    # Split into train and test sets
    # MNIST has 60,000 training images and 10,000 test images
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print(
        f"MNIST dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples"
    )

    return X_train, y_train, X_test, y_test


def main():
    """Main function to train and evaluate a model on MNIST."""
    print("Loading MNIST dataset...")

    # Load MNIST data
    train_images, train_labels, test_images, test_labels = load_mnist_data()

    # For faster training in this example, we'll use a subset
    # Comment these lines if you want to use the full dataset
    subset_size = 10000
    val_size = 2000

    indices = np.random.permutation(len(train_images))
    train_idx, val_idx = indices[val_size : subset_size + val_size], indices[:val_size]

    train_data = train_images[train_idx]
    train_targets = train_labels[train_idx]
    val_data = train_images[val_idx]
    val_targets = train_labels[val_idx]

    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

    # Create datasets
    train_dataset = Dataset(train_data, train_targets)
    val_dataset = Dataset(val_data, val_targets)

    # Create dataloaders with reasonable batch sizes
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = Sequential(
        Linear(784, 128),
        BatchNorm(128),
        ReLU(),
        Dropout(0.3),
        Linear(128, 64),
        BatchNorm(64),
        ReLU(),
        Dropout(0.3),
        Linear(64, 10),
        # Softmax(),
    )

    # Print model architecture summary
    model.summary((784,))

    # Create loss function and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Create learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Create tracker with early stopping
    tracker = TrainingTracker(
        experiment_name="mnist_full",
        early_stopping={"patience": 5, "metric": "val_loss", "min_delta": 0.001},
    )

    # Train model
    print("\nStarting training...")
    tracker = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=20,  # Increased epochs for better learning
        scheduler=scheduler,
        tracker=tracker,
    )

    # Show final summary
    tracker.summary(show_best=True)

    # Plot training metrics
    tracker.plot(save_path="mnist_training_metrics.png")
    print(f"Training plot saved to mnist_training_metrics.png")

    # Save model
    model_path = "mnist_model.pkl"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    # Load model (demonstration)
    loaded_model = load_model(model_path)

    # Evaluate on test set (use a subset for quick demonstration)
    test_subset_size = 1000
    test_indices = np.random.choice(len(test_images), test_subset_size, replace=False)
    test_data = test_images[test_indices]
    test_labels = test_labels[test_indices]

    test_dataset = Dataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\nEvaluating on test set...")
    test_metrics = evaluate(loaded_model, test_loader, loss_fn)
    print("Test metrics:", test_metrics)

    # Compute class-wise accuracy (if possible)
    print("\nCalculating per-class accuracy...")
    try:
        # This requires storing predictions and targets
        class_correct = np.zeros(10)
        class_total = np.zeros(10)

        for x, y in test_loader:
            outputs = loaded_model(x)
            predicted = np.argmax(outputs.data, axis=1)
            predicted = predicted.reshape(-1)

            # Update class-wise accuracy
            for i in range(len(y.data)):
                label = int(y.data[i])
                class_correct[label] += predicted[i] == label
                class_total[label] += 1

        # Print per-class accuracy
        for i in range(10):
            if class_total[i] > 0:
                print(
                    f"Digit {i}: {100 * class_correct[i] / class_total[i]:.2f}% accuracy"
                )

    except Exception as e:
        print(f"Could not compute per-class accuracy: {e}")

    print("\nMNIST example completed successfully!")


if __name__ == "__main__":
    main()
