"""
Stable XOR problem solution using Spectral Normalization.

This example demonstrates how to solve the XOR problem reliably using
Spectral Normalization to improve training stability and convergence.
"""

import numpy as np
import matplotlib.pyplot as plt

from core.tensor import Tensor
from nn.activations import Tanh
from nn.linear import Linear
from nn.sequential import Sequential
from nn.spectral_norm import SpectralNormLinear
from nn.normalization import BatchNorm
from train.loss import MSELoss
from train.optim import Adam
from monitor.tracker import TrainingTracker


def plot_decision_boundary(model, title="Decision Boundary"):
    """Plot the decision boundary of the trained model."""
    h = 0.01  # Step size for mesh grid
    x_min, x_max = -0.2, 1.2
    y_min, y_max = -0.2, 1.2

    # Create mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # Make predictions on mesh grid
    Z = []
    for point in mesh_points:
        # Convert point to tensor
        x_point = Tensor(point.reshape(1, -1))
        # Forward pass
        output = model(x_point)
        # Get prediction (greater than 0.5 is class 1)
        pred = 1 if output.data[0][0] >= 0.5 else 0
        Z.append(pred)

    # Reshape predictions to match mesh grid
    Z = np.array(Z).reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 8))

    # Plot contour with filled regions
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)

    # Plot decision boundary line (0.5 threshold)
    plt.contour(xx, yy, Z, levels=[0.5], colors="k", linestyles="-", linewidths=2)

    # XOR data points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors="k", s=100)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.savefig("xor_spectral_norm_solution.png")
    plt.show()


def solve_xor_with_spectral_norm():
    """Solve the XOR problem using spectral normalization for stability."""
    print("Solving XOR problem with Spectral Normalization...")

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Convert to tensors
    X_tensor = Tensor(X, requires_grad=True)
    y_tensor = Tensor(y, requires_grad=False)

    # Create a model with Spectral Normalization
    # Using SpectralNormLinear for the first layer helps stabilize training
    model = Sequential(
        SpectralNormLinear(2, 8),  # Spectrally normalized layer
        BatchNorm(8),  # Maybe BatchNorm would work here?
        Tanh(),
        Linear(8, 1),  # Regular layer for output
    )

    # Create loss function
    loss_fn = MSELoss()

    # Create optimizer with appropriate learning rate
    optimizer = Adam(model.parameters(), lr=0.05)

    # Create tracker
    tracker = TrainingTracker(experiment_name="xor_spectral_norm")

    # Training loop
    epochs = 2000

    for epoch in range(1, epochs + 1):
        # Start tracking epoch
        tracker.start_epoch()

        # Forward pass
        outputs = model(X_tensor)
        loss = loss_fn(outputs, y_tensor)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        # Compute accuracy
        predictions = []
        for i, x in enumerate(X):
            x_input = Tensor(x.reshape(1, -1))
            output = model(x_input)
            pred = 1 if output.data[0][0] >= 0.5 else 0
            predictions.append(pred)

        accuracy = np.mean(np.array(predictions) == y.flatten()) * 100

        # Log metrics
        tracker.log(loss=loss.data, acc=accuracy / 100)  # Normalize accuracy to [0,1]

        # Print progress
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}/{epochs}, Loss: {loss.data:.4f}, Accuracy: {accuracy:.2f}%"
            )

        # Early stopping if we reach 100% accuracy
        if accuracy == 100.0 and epoch > 500:
            print(f"Reached 100% accuracy at epoch {epoch}. Early stopping.")
            break

    # Plot training metrics
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(tracker.logs["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(np.array(tracker.logs["acc"]) * 100)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("xor_spectral_norm_training.png")

    # Final evaluation
    print("\nFinal Results:")
    print("XOR Problem Predictions:")

    for i, x in enumerate(X):
        x_input = Tensor(x.reshape(1, -1))
        output = model(x_input)
        pred_value = output.data[0][0]
        pred_class = 1 if pred_value >= 0.5 else 0
        print(
            f"Input: {x}, Target: {y[i][0]}, Prediction: {pred_value:.4f} -> {pred_class}"
        )

    # Calculate overall accuracy
    final_outputs = model(X_tensor).data
    final_predictions = (final_outputs >= 0.5).astype(int)
    final_accuracy = np.mean(final_predictions.flatten() == y.flatten()) * 100

    print(f"\nFinal Accuracy: {final_accuracy:.2f}%")

    # Plot decision boundary
    plot_decision_boundary(
        model, title="XOR Decision Boundary with Spectral Normalization"
    )

    print("\nTraining plot saved to 'xor_spectral_norm_training.png'")
    print("Decision boundary saved to 'xor_spectral_norm_solution.png'")

    return model, tracker


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Solve XOR problem with spectral normalization
    model, tracker = solve_xor_with_spectral_norm()
