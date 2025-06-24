"""
This example demonstrates how to use the Sharpness-Aware Minimization (SAM) optimizer
for improved generalization and robustness on image classification tasks.

We'll compare standard optimizers against SAM on the CIFAR-10 dataset
to show the generalization benefits.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fit.core.tensor import Tensor
from fit.nn.modules.activation import ReLU
from fit.nn.modules.linear import Linear
from fit.nn.modules.container import Sequential
from fit.nn.modules.normalization import BatchNorm
from fit.loss.classification import CrossEntropyLoss
from fit.optim.sgd import SGD
from fit.optim.adam import Adam
from fit.optim.experimental.sam import SAM
from fit.data.dataset import Dataset
from fit.data.dataloader import DataLoader


def load_cifar10_subset(n_samples=10000):
    """
    Load a subset of CIFAR-10 data for faster experimentation.

    Args:
        n_samples: Number of samples to load

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("Loading CIFAR-10 dataset...")

    # Load data from OpenML
    cifar = fetch_openml(name="CIFAR_10", version=1, parser="auto", as_frame=False)
    X = cifar.data.astype("float32") / 255.0
    y = cifar.target.astype("int")

    # Convert features to grayscale to simplify
    # Shape from (n, 3072) to (n, 1024) by averaging RGB channels
    X_reshaped = X.reshape(-1, 3, 32, 32)
    X_gray = X_reshaped.mean(axis=1).reshape(-1, 1024)

    # Take subset for faster training
    indices = np.random.permutation(len(X_gray))[:n_samples]
    X_subset = X_gray[indices]
    y_subset = y[indices]

    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_subset, test_size=0.3, random_state=42, stratify=y_subset
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(
        f"Dataset ready: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model():
    """
    Create a simple CNN-like model for CIFAR-10 classification.

    Returns:
        Sequential model
    """
    model = Sequential(
        Linear(1024, 256),
        ReLU(),
        BatchNorm(256),
        Linear(256, 128),
        ReLU(),
        BatchNorm(128),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),  # 10 classes for CIFAR-10
    )

    return model


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=20):
    """
    Train a model and track performance.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        criterion: Loss function
        epochs: Number of epochs

    Returns:
        Dictionary with training history
    """
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "train_time": []}

    print(f"Starting training with {optimizer.__class__.__name__}...")

    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # Zero gradients
            for param in model.parameters():
                param.grad = None

            # Forward pass
            output = model(batch_x)
            loss = criterion(output, batch_y)

            # Backward pass
            loss.backward()

            # SAM specific step
            if isinstance(optimizer, SAM):
                optimizer.first_step(zero_grad=True)

                # Second forward pass for SAM
                output2 = model(batch_x)
                loss2 = criterion(output2, batch_y)
                loss2.backward()

                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()

            epoch_train_loss += loss.data
            train_batches += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.data:.4f}")

        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)

        # Record metrics
        avg_train_loss = epoch_train_loss / train_batches
        epoch_time = time.time() - start_time

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["train_time"].append(epoch_time)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print("-" * 50)

    return history


def evaluate_model(model, data_loader, criterion):
    """
    Evaluate model on validation/test data.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    for batch_x, batch_y in data_loader:
        output = model(batch_x)
        loss = criterion(output, batch_y)

        total_loss += loss.data
        batches += 1

        # Calculate accuracy
        predictions = np.argmax(output.data, axis=1)
        targets = batch_y.data if hasattr(batch_y, "data") else batch_y

        correct += np.sum(predictions == targets)
        total += len(targets)

    avg_loss = total_loss / batches
    accuracy = correct / total

    return avg_loss, accuracy


def compare_optimizers():
    """
    Compare different optimizers on CIFAR-10 subset.
    """
    print("SAM vs Standard Optimizers Comparison")
    print("=" * 50)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_cifar10_subset(n_samples=5000)

    # Create data loaders
    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)
    test_dataset = Dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Loss function
    criterion = CrossEntropyLoss()

    # Define optimizers to compare
    optimizers_config = [
        ("SGD", lambda params: SGD(params, lr=0.01, momentum=0.9)),
        ("Adam", lambda params: Adam(params, lr=0.001)),
        (
            "SAM-SGD",
            lambda params: SAM(params, SGD(params, lr=0.01, momentum=0.9), rho=0.05),
        ),
        ("SAM-Adam", lambda params: SAM(params, Adam(params, lr=0.001), rho=0.05)),
    ]

    results = {}

    for name, optimizer_fn in optimizers_config:
        print(f"\n{'='*20} Training with {name} {'='*20}")

        # Create fresh model for each optimizer
        model = create_model()
        optimizer = optimizer_fn(model.parameters())

        # Train model
        history = train_model(
            model, train_loader, val_loader, optimizer, criterion, epochs=10
        )

        # Final test evaluation
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)

        results[name] = {
            "history": history,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }

        print(f"\nFinal {name} Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot comparison
    plot_comparison(results)

    return results


def plot_comparison(results):
    """
    Plot comparison of different optimizers.

    Args:
        results: Dictionary with results from each optimizer
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot training loss
        axes[0, 0].set_title("Training Loss")
        for name, data in results.items():
            axes[0, 0].plot(data["history"]["train_loss"], label=name)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot validation loss
        axes[0, 1].set_title("Validation Loss")
        for name, data in results.items():
            axes[0, 1].plot(data["history"]["val_loss"], label=name)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot validation accuracy
        axes[1, 0].set_title("Validation Accuracy")
        for name, data in results.items():
            axes[1, 0].plot(data["history"]["val_accuracy"], label=name)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot final test accuracy comparison
        axes[1, 1].set_title("Final Test Accuracy")
        names = list(results.keys())
        test_accuracies = [results[name]["test_accuracy"] for name in names]

        bars = axes[1, 1].bar(names, test_accuracies)
        axes[1, 1].set_ylabel("Test Accuracy")
        axes[1, 1].set_ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, test_accuracies):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig("sam_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("Comparison plot saved as 'sam_comparison.png'")

    except Exception as e:
        print(f"Could not generate plot: {e}")


def demonstrate_sam_benefits():
    """
    Demonstrate the key benefits of SAM optimizer.
    """
    print("\nSAM Benefits Demonstration:")
    print("=" * 40)
    print("1. Better Generalization: SAM typically achieves better test accuracy")
    print("2. Flatter Minima: SAM finds parameters in flatter loss landscapes")
    print("3. Robustness: More stable to hyperparameter choices")
    print("4. Reduced Overfitting: Better gap between train and validation performance")
    print("\nKey SAM Parameters:")
    print("- rho: Controls the sharpness penalty (typical: 0.05-0.2)")
    print("- base_optimizer: Underlying optimizer (SGD, Adam, etc.)")
    print("- adaptive: Whether to use adaptive rho (default: False)")


if __name__ == "__main__":
    print("Sharpness-Aware Minimization (SAM) Example")
    print("=" * 50)

    # Show SAM benefits explanation
    demonstrate_sam_benefits()

    # Ask user if they want to run the comparison
    choice = input("\nRun optimizer comparison? (y/n): ").lower()

    if choice == "y":
        try:
            results = compare_optimizers()

            print("\n" + "=" * 50)
            print("SUMMARY")
            print("=" * 50)

            # Find best performing optimizer
            best_optimizer = max(
                results.keys(), key=lambda x: results[x]["test_accuracy"]
            )
            best_accuracy = results[best_optimizer]["test_accuracy"]

            print(f"Best performing optimizer: {best_optimizer}")
            print(f"Best test accuracy: {best_accuracy:.4f}")

            # Show SAM vs non-SAM comparison
            sam_optimizers = [name for name in results.keys() if "SAM" in name]
            regular_optimizers = [name for name in results.keys() if "SAM" not in name]

            if sam_optimizers and regular_optimizers:
                avg_sam_acc = np.mean(
                    [results[name]["test_accuracy"] for name in sam_optimizers]
                )
                avg_regular_acc = np.mean(
                    [results[name]["test_accuracy"] for name in regular_optimizers]
                )

                print(f"\nAverage SAM accuracy: {avg_sam_acc:.4f}")
                print(f"Average regular accuracy: {avg_regular_acc:.4f}")
                print(f"SAM improvement: {avg_sam_acc - avg_regular_acc:.4f}")

        except Exception as e:
            print(f"Error running comparison: {e}")
            import traceback

            traceback.print_exc()

    else:
        print("Skipping comparison. To run it later, execute this script again.")

    print("\nSAM example completed!")
