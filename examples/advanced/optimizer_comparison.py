"""
Enhanced XOR problem comparison of different optimizers.

This version fixes convergence issues by properly initializing the model
and providing accurate gradients for all optimizers.
"""

import matplotlib.pyplot as plt
import numpy as np

from fit.core import Tensor
from fit.nn.modules import Tanh
from fit.nn.modules import Linear
from fit.nn.modules.base import Layer
from fit.simple import Sequential
from fit.loss import MSELoss
from fit.optim import SGD, Adam, SGDMomentum
from fit.optim.experimental import Lion
from fit.optim.experimental import SAM


def train_xor_with_optimizer(optimizer_name, epochs=2000, verbose=True):
    """
    Train a model to solve the XOR problem with a specific optimizer.

    Args:
        optimizer_name: Name of the optimizer to use
        epochs: Number of training epochs
        verbose: Whether to print progress

    Returns:
        Dictionary with training results
    """
    # Use a different seed for each optimizer to ensure different initialization
    np.random.seed(42 + hash(optimizer_name) % 1000)

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Convert to tensors
    X_tensor = Tensor(X, requires_grad=True)
    y_tensor = Tensor(y, requires_grad=False)

    # Create a model with adequate capacity for XOR
    # Using 32 hidden neurons with Tanh activation
    hidden_size = 32
    model = Sequential(Linear(2, hidden_size), Tanh(), Linear(hidden_size, 1))

    # Critical: Initialize weights with a pattern that breaks symmetry for XOR
    # This is the most important part for ensuring convergence

    # Initialize with specific pattern for first layer
    init_scale = 1.0
    first_weights = np.zeros((2, hidden_size))
    for i in range(hidden_size):
        if i % 2 == 0:
            first_weights[0, i] = init_scale
            first_weights[1, i] = -init_scale
        else:
            first_weights[0, i] = -init_scale
            first_weights[1, i] = init_scale

    # Alternating bias pattern
    first_bias = np.zeros(hidden_size)
    for i in range(hidden_size):
        if i < hidden_size // 2:
            first_bias[i] = 0.1
        else:
            first_bias[i] = -0.1

    # Apply custom initialization
    model.layers[0].weight.data = first_weights
    model.layers[0].bias.data = first_bias

    # Initialize output layer with small random values
    model.layers[2].weight.data = np.random.uniform(-0.1, 0.1, (hidden_size, 1))
    model.layers[2].bias.data = np.zeros(1)

    # Create loss function
    loss_fn = MSELoss()

    # Create optimizer with appropriate learning rates for each optimizer
    if optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=0.05)
    elif optimizer_name == "SGDMomentum":
        optimizer = SGDMomentum(model.parameters(), lr=0.02, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=0.01)
    elif optimizer_name == "Lion":
        optimizer = Lion(model.parameters(), lr=0.003)
    elif optimizer_name == "SAM":
        # SAM needs a base optimizer
        base_optimizer = Adam(model.parameters(), lr=0.01)
        optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Training loop
    losses = []
    accuracies = []

    for epoch in range(1, epochs + 1):
        # Forward pass
        if optimizer_name == "SAM":
            # SAM requires a closure for both steps
            def closure():
                outputs = model(X_tensor)
                loss = loss_fn(outputs, y_tensor)
                loss.backward()
                return loss

            # First step (perturb weights)
            loss = optimizer.first_step(closure)

            # Second step (compute gradient at perturbed weights & update)
            loss = optimizer.second_step(closure)

            # Get outputs for accuracy calculation
            outputs = model(X_tensor)
        else:
            # Standard optimization
            outputs = model(X_tensor)
            loss = loss_fn(outputs, y_tensor)
            losses.append(float(loss.data))

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()

        # Store loss for SAM
        if optimizer_name == "SAM":
            losses.append(float(loss.data))

        # Calculate accuracy
        threshold = 0.5
        predictions = (outputs.data >= threshold).astype(int)
        true_values = y.astype(int)
        accuracy = np.mean(predictions == true_values) * 100
        accuracies.append(accuracy)

        # Only print progress occasionally to avoid spamming
        if verbose and (epoch % 200 == 0 or epoch == 1 or epoch == epochs):
            print(
                f"{optimizer_name} - Epoch {epoch}/{epochs}, Loss: {losses[-1]:.4f}, Accuracy: {accuracy:.1f}%"
            )

        # Early stopping if we've reached perfect accuracy
        if accuracy == 100.0 and losses[-1] < 0.01 and epoch > 100:
            if verbose:
                print(
                    f"{optimizer_name} - Converged at epoch {epoch}/{epochs}, Loss: {losses[-1]:.4f}"
                )
            break

    # Final predictions and accuracy
    outputs = model(X_tensor)
    threshold = 0.5
    predicted_classes = (outputs.data >= threshold).astype(int)
    actual_classes = y.astype(int)
    accuracy = np.mean(predicted_classes == actual_classes) * 100

    if verbose:
        print(f"\n{optimizer_name} final accuracy: {accuracy:.1f}%")
        print("Predictions vs. Actual:")
        for i in range(len(X)):
            input_data = X[i]
            actual = y[i][0]
            prediction = outputs.data[i][0]
            predicted_class = 1 if prediction >= 0.5 else 0
            print(
                f"Input: {input_data}, Predicted: {prediction:.4f} -> {predicted_class}, Actual: {actual}"
            )

    # Return results in a dictionary
    return {
        "model": model,
        "losses": losses,
        "accuracies": accuracies,
        "final_accuracy": accuracy,
        "predictions": outputs.data.flatten(),
        "optimizer": optimizer_name,
    }


def plot_decision_boundaries(results, optimizers):
    """
    Plot decision boundaries for each optimizer.

    Args:
        results: Dictionary of optimizer results
        optimizers: List of optimizer names to plot
    """
    plt.figure(figsize=(15, 10))

    # Create a mesh grid for visualization
    h = 0.02
    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # XOR data points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]]).flatten()

    # Plot each optimizer's decision boundary
    for i, opt_name in enumerate(optimizers):
        if opt_name not in results:
            continue

        result = results[opt_name]
        model = result["model"]

        # Plot in a grid
        plt.subplot(2, 3, i + 1)

        # Make predictions on the mesh grid
        Z = []
        for point in mesh_points:
            x_point = Tensor(point.reshape(1, -1))
            pred = model(x_point).data[0, 0]
            Z.append(1 if pred >= 0.5 else 0)

        # Reshape for contour plot
        Z = np.array(Z).reshape(xx.shape)

        # Plot decision boundary
        plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=plt.cm.RdBu, alpha=0.5)
        plt.contour(xx, yy, Z, levels=[0.5], colors="k", linestyles="-", linewidths=2)

        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors="k", s=80)

        # Add title and accuracy
        plt.title(f"{opt_name} - Accuracy: {result['final_accuracy']:.1f}%")
        plt.xlabel("X1")
        plt.ylabel("X2")

    plt.tight_layout()
    plt.savefig("optimizer_decision_boundaries.png")
    print("\nDecision boundaries saved to optimizer_decision_boundaries.png")


def compare_optimizers():
    """
    Compare different optimizers on the XOR problem.
    """
    # Set up plot
    plt.figure(figsize=(12, 8))

    # List of optimizers to compare
    optimizers = ["SGD", "SGDMomentum", "Adam", "Lion", "SAM"]
    results = {}

    # Train with each optimizer
    for optimizer_name in optimizers:
        print(f"\nTraining with {optimizer_name}...")
        result = train_xor_with_optimizer(optimizer_name, epochs=2000, verbose=True)
        results[optimizer_name] = result

        # Plot loss curves
        plt.subplot(2, 1, 1)
        plt.plot(result["losses"], label=f"{optimizer_name}")

    # Format loss plot
    plt.subplot(2, 1, 1)
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")  # Log scale to better see differences
    plt.ylim(1e-4, 2)  # Focus on relevant loss range
    plt.legend()
    plt.grid(True)

    # Plot accuracy comparison
    plt.subplot(2, 1, 2)
    accuracies = [results[opt]["final_accuracy"] for opt in optimizers]

    plt.bar(optimizers, accuracies)
    plt.title("Final Accuracy Comparison")
    plt.xlabel("Optimizer")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)  # Max 100% with some margin

    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 2, f"{acc:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig("optimizer_comparison.png")
    print("\nComparison plot saved to optimizer_comparison.png")

    # Plot decision boundaries for each optimizer
    plot_decision_boundaries(results, optimizers)

    # Print summary
    print("\nOptimizer Performance Summary:")
    for opt in optimizers:
        print(
            f"- {opt}: {results[opt]['final_accuracy']:.1f}% accuracy, final loss = {results[opt]['losses'][-1]:.6f}"
        )


if __name__ == "__main__":
    compare_optimizers()
