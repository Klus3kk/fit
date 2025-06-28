"""
Enhanced XOR problem comparison of different optimizers.

This version fixes convergence issues by properly initializing the model
and providing accurate gradients for all optimizers.
"""

import matplotlib.pyplot as plt
import numpy as np

from fit.core.tensor import Tensor
from fit.nn.modules.activation import Tanh
from fit.nn.modules.linear import Linear
from fit.nn.modules.container import Sequential
from fit.loss.regression import MSELoss
from fit.optim.sgd import SGD, SGDMomentum
from fit.optim.adam import Adam
from fit.optim.experimental.sam import SAM


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
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

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
    first_weights = np.zeros((hidden_size, 2))  # Fixed: proper shape for weight matrix
    for i in range(hidden_size):
        if i % 2 == 0:
            first_weights[i, 0] = init_scale
            first_weights[i, 1] = -init_scale
        else:
            first_weights[i, 0] = -init_scale
            first_weights[i, 1] = init_scale

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

    # Initialize second layer with smaller random weights
    model.layers[2].weight.data = np.random.normal(0, 0.1, model.layers[2].weight.data.shape)
    model.layers[2].bias.data = np.zeros_like(model.layers[2].bias.data)

    # Set up optimizer
    if optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=0.1)
    elif optimizer_name == "SGDMomentum":
        optimizer = SGDMomentum(model.parameters(), lr=0.1, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=0.01)
    elif optimizer_name == "SAM":
        # Use SGDMomentum as base optimizer for SAM, not SGD with momentum
        base_optimizer = SGDMomentum(model.parameters(), lr=0.1, momentum=0.9)
        optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Loss function
    loss_fn = MSELoss()

    # Track training progress
    losses = []
    accuracies = []

    # Training loop
    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()

        if optimizer_name == "SAM":
            # SAM requires two forward passes
            outputs = model(X_tensor)
            loss = loss_fn(outputs, y_tensor)
            loss.backward()
            
            # First step: perturb weights
            optimizer.first_step(zero_grad=True)
            
            # Second forward pass
            outputs = model(X_tensor)
            loss = loss_fn(outputs, y_tensor)
            loss.backward()
            
            # Second step: update weights
            optimizer.second_step(zero_grad=True)
        else:
            # Standard optimization
            outputs = model(X_tensor)
            loss = loss_fn(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        losses.append(float(loss.data))

        # Calculate accuracy
        threshold = 0.5
        predictions = (outputs.data >= threshold).astype(int)
        true_values = y.astype(int)
        accuracy = np.mean(predictions == true_values) * 100
        accuracies.append(accuracy)

        # Print progress occasionally
        if verbose and (epoch % 200 == 0 or epoch == 1 or epoch == epochs-1):
            print(f"{optimizer_name} - Epoch {epoch}/{epochs}, Loss: {losses[-1]:.4f}, Accuracy: {accuracy:.1f}%")

        # Early stopping if converged
        if accuracy == 100.0 and losses[-1] < 0.01 and epoch > 100:
            if verbose:
                print(f"{optimizer_name} - Converged at epoch {epoch}/{epochs}")
            break

    # Final evaluation
    outputs = model(X_tensor)
    threshold = 0.5
    predicted_classes = (outputs.data >= threshold).astype(int)
    actual_classes = y.astype(int)
    final_accuracy = np.mean(predicted_classes == actual_classes) * 100

    if verbose:
        print(f"\n{optimizer_name} final results:")
        print(f"  Final accuracy: {final_accuracy:.1f}%")
        print(f"  Final loss: {losses[-1]:.4f}")

    return {
        "model": model,
        "losses": losses,
        "accuracies": accuracies,
        "final_accuracy": final_accuracy,
        "predictions": outputs.data.flatten(),
        "optimizer": optimizer_name,
    }


def plot_decision_boundaries(results, optimizers):
    """
    Plot decision boundaries for each optimizer.
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
            x_point = Tensor(point.reshape(1, -1), requires_grad=False)
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
    plt.savefig("optimizer_decision_boundaries.png", dpi=150, bbox_inches="tight")
    print("\nDecision boundaries saved to optimizer_decision_boundaries.png")


def compare_optimizers():
    """
    Compare different optimizers on the XOR problem.
    """
    print("Optimizer Comparison on XOR Problem")
    print("=" * 40)

    # List of optimizers to compare
    optimizers = ["SGD", "SGDMomentum", "Adam", "SAM"]
    results = {}

    # Train with each optimizer
    for optimizer_name in optimizers:
        print(f"\nTraining with {optimizer_name}...")
        try:
            result = train_xor_with_optimizer(optimizer_name, epochs=1000, verbose=True)
            results[optimizer_name] = result
        except Exception as e:
            print(f"Error training with {optimizer_name}: {e}")
            continue

    if not results:
        print("No successful training runs!")
        return

    # Set up plot
    plt.figure(figsize=(12, 8))

    # Plot loss curves
    plt.subplot(2, 1, 1)
    for optimizer_name, result in results.items():
        plt.plot(result["losses"], label=f"{optimizer_name}")

    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.ylim(1e-4, 2)
    plt.legend()
    plt.grid(True)

    # Plot accuracy comparison
    plt.subplot(2, 1, 2)
    accuracies = [results[opt]["final_accuracy"] for opt in results.keys()]
    optimizer_names = list(results.keys())

    plt.bar(optimizer_names, accuracies)
    plt.title("Final Accuracy Comparison")
    plt.xlabel("Optimizer")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)

    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 2, f"{acc:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig("optimizer_comparison.png", dpi=150, bbox_inches="tight")
    print("\nComparison plot saved to optimizer_comparison.png")

    # Plot decision boundaries
    plot_decision_boundaries(results, optimizer_names)

    # Print summary
    print("\nOptimizer Performance Summary:")
    for opt in optimizer_names:
        print(f"- {opt}: {results[opt]['final_accuracy']:.1f}% accuracy, final loss = {results[opt]['losses'][-1]:.6f}")


if __name__ == "__main__":
    compare_optimizers()