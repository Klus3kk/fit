"""
XOR problem solver that works correctly.

This module demonstrates how to solve the XOR problem with a neural network.
This version addresses the convergence issue seen in the examples.
"""

import numpy as np
import matplotlib.pyplot as plt

from core.tensor import Tensor
from nn.linear import Linear
from nn.activations import Tanh
from nn.sequential import Sequential
from train.loss import MSELoss
from train.optim import Adam


def solve_xor():
    """
    Solve the XOR problem reliably.

    This solution specifically addresses the convergence issues
    in the existing implementation.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Convert to tensors
    X_tensor = Tensor(X, requires_grad=True)
    y_tensor = Tensor(y, requires_grad=False)

    # Create model with a larger hidden layer and asymmetric initialization
    # to break symmetry - critical for XOR problem
    model = Sequential(Linear(2, 8), Tanh(), Linear(8, 1))

    # Manual initialization - crucial for solving XOR
    # This creates asymmetric weights
    model.layers[0].weight.data = np.random.uniform(-1, 1, (2, 8))
    model.layers[0].bias.data = np.random.uniform(-0.1, 0.1, 8)

    # Create loss function and optimizer
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.03)

    # Training loop
    losses = []
    epochs = 1000

    # Early stopping variables
    best_loss = float("inf")
    patience = 50
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Forward pass
        outputs = model(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        losses.append(loss.data)

        # Early stopping check
        if loss.data < best_loss:
            best_loss = loss.data
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and loss.data < 0.1:
            print(f"Early stopping at epoch {epoch} with loss: {loss.data:.4f}")
            break

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        # Print progress
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.data:.4f}")

    # Plot loss curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    # Plot decision boundary
    plt.subplot(1, 2, 2)

    # Create a mesh grid for visualization
    h = 0.01
    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Make predictions on the grid
    Z = []
    for point in grid_points:
        x_point = Tensor(point.reshape(1, -1))
        pred = model(x_point).data[0, 0]
        Z.append(pred)

    Z = np.array(Z).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.RdBu_r, edgecolors="k")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Decision Boundary")

    plt.tight_layout()
    plt.savefig("xor_solved.png")
    plt.show()

    # Test the model
    with_threshold = lambda x: 1 if x >= 0.5 else 0

    print("\nPredictions vs. Actual:")
    for i in range(len(X)):
        input_data = X[i]
        actual = y[i][0]

        # Make prediction
        x_input = Tensor(input_data.reshape(1, -1))
        prediction = model(x_input).data[0][0]
        predicted_class = with_threshold(prediction)

        print(
            f"Input: {input_data}, Predicted: {prediction:.4f} -> {predicted_class}, Actual: {actual}"
        )

    # Calculate accuracy
    outputs = model(X_tensor).data
    predicted_classes = np.array([with_threshold(x[0]) for x in outputs])
    actual_classes = y.flatten()
    accuracy = np.mean(predicted_classes == actual_classes) * 100

    print(f"\nAccuracy: {accuracy:.2f}%")

    return model, accuracy


if __name__ == "__main__":
    solve_xor()
