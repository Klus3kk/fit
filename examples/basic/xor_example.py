"""
Minimal version of XOR example that avoids the Mean error.

This implementation works with the existing codebase by avoiding
the problematic mean operation in the MSELoss function.
"""

import numpy as np
import matplotlib.pyplot as plt

from fit.core.tensor import Tensor
from fit.nn.modules.activation import Tanh
from fit.nn.modules.linear import Linear
from fit.nn.modules.container import Sequential
from fit.optim.adam import Adam


def custom_mse_loss(predictions, targets):
    """
    Custom MSE loss that avoids using the mean() operation directly.
    Instead calculates MSE manually to avoid the error.
    """
    # Calculate squared error
    diff = predictions.data - targets.data
    squared_diff = diff * diff

    # Calculate mean manually
    mse = np.sum(squared_diff) / squared_diff.size

    # Create loss tensor with gradient
    loss = Tensor(mse, requires_grad=True)

    # Define gradient for backward pass
    def _backward_fn():
        # Calculate gradient of MSE: 2 * (pred - target) / n
        grad = 2 * diff / squared_diff.size
        predictions.grad = grad if predictions.grad is None else predictions.grad + grad

    # Set up backward function
    loss.backward_fn = _backward_fn

    return loss


def solve_xor():
    """
    Solve the XOR problem using a custom loss function that avoids the error.
    """
    print("Solving XOR problem with neural network...")

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    print("Dataset:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {y[i][0]}")

    # Create network: 2 -> 3 -> 1 with Tanh activations
    model = Sequential(Linear(2, 3), Tanh(), Linear(3, 1), Tanh())

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.1)

    # Training loop
    epochs = 1000
    losses = []

    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(X)):
            # Convert to tensors
            x_tensor = Tensor([X[i]], requires_grad=True)
            y_tensor = Tensor([y[i]], requires_grad=False)

            # Forward pass
            output = model(x_tensor)

            # Calculate loss using custom MSE
            loss = custom_mse_loss(output, y_tensor)

            # Zero gradients
            for param in model.parameters():
                param.grad = None

            # Backward pass
            try:
                if hasattr(loss, "backward_fn"):
                    loss.backward_fn()
                else:
                    loss.backward()

                # Update parameters
                optimizer.step()

                total_loss += loss.data
            except Exception as e:
                print(f"Error in training at epoch {epoch}: {e}")
                return False

        # Record average loss
        avg_loss = total_loss / len(X)
        losses.append(avg_loss)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

    # Test the trained model
    print("\nTesting trained model:")
    print("Input -> Output (Target)")
    for i in range(len(X)):
        x_tensor = Tensor([X[i]], requires_grad=False)
        output = model(x_tensor)
        predicted = output.data[0][0]
        actual = y[i][0]
        print(f"{X[i]} -> {predicted:.4f} ({actual})")

    # Check if learning was successful
    print("\nEvaluating results:")
    correct = 0
    for i in range(len(X)):
        x_tensor = Tensor([X[i]], requires_grad=False)
        output = model(x_tensor)
        predicted = 1 if output.data[0][0] > 0.5 else 0
        actual = int(y[i][0])
        if predicted == actual:
            correct += 1
        print(
            f"Input: {X[i]}, Predicted: {predicted}, Actual: {actual}, {'✓' if predicted == actual else '✗'}"
        )

    accuracy = correct / len(X)
    print(f"\nAccuracy: {accuracy:.2%}")

    # Plot loss curve
    try:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        # Create a grid to visualize the decision boundary
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 50), np.linspace(-0.5, 1.5, 50))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Get model predictions for the grid
        Z = []
        for point in grid_points:
            x_tensor = Tensor([point], requires_grad=False)
            output = model(x_tensor)
            Z.append(output.data[0][0])

        Z = np.array(Z).reshape(xx.shape)

        # Plot decision boundary
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap="RdYlBu")
        plt.colorbar(label="Output")

        # Plot data points
        colors = ["red", "blue"]
        for i in range(len(X)):
            plt.scatter(
                X[i][0], X[i][1], c=colors[int(y[i][0])], s=100, edgecolors="black"
            )

        plt.title("XOR Decision Boundary")
        plt.xlabel("Input 1")
        plt.ylabel("Input 2")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("xor_results.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("Plot saved as 'xor_results.png'")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    return accuracy > 0.75  # Return True if we got at least 75% accuracy


if __name__ == "__main__":
    success = solve_xor()
    if success:
        print("\n🎉 XOR problem solved successfully!")
    else:
        print("\n❌ Failed to solve XOR problem")
