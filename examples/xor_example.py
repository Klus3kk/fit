"""
Minimal version of XOR example that avoids the Mean error.

This implementation works with the existing codebase by avoiding
the problematic mean operation in the MSELoss function.
"""

import numpy as np
import matplotlib.pyplot as plt

from core.tensor import Tensor
from nn.activations import Tanh
from nn.linear import Linear
from nn.sequential import Sequential
from train.optim import Adam


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
    # Set random seed for reproducibility
    np.random.seed(42)

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Convert to tensors
    X_tensor = Tensor(X, requires_grad=True)
    y_tensor = Tensor(y, requires_grad=False)

    # Create model with a proper hidden layer size
    hidden_size = 32
    model = Sequential(Linear(2, hidden_size), Tanh(), Linear(hidden_size, 1))

    # Initialize first layer with specific pattern
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

    # Apply initialization
    model.layers[0].weight.data = first_weights
    model.layers[0].bias.data = first_bias

    # Initialize output layer
    model.layers[2].weight.data = np.random.uniform(-0.1, 0.1, (hidden_size, 1))
    model.layers[2].bias.data = np.zeros(1)

    # Create optimizer (no loss function, we'll use custom_mse_loss)
    optimizer = Adam(model.parameters(), lr=0.01)  # Lower learning rate for stability

    # Training loop
    losses = []
    accuracies = []
    epochs = 2000

    print("Training XOR model...")
    for epoch in range(1, epochs + 1):
        # Forward pass
        outputs = model(X_tensor)

        # Use custom loss function to avoid the error
        loss = custom_mse_loss(outputs, y_tensor)
        losses.append(loss.data)

        # Calculate accuracy
        threshold = 0.5
        predictions = (outputs.data >= threshold).astype(int)
        true_values = y.astype(int)
        accuracy = np.mean(predictions == true_values) * 100
        accuracies.append(accuracy)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        # Print progress
        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{epochs}, Loss: {loss.data:.4f}, Accuracy: {accuracy:.1f}%"
            )

        # Early stopping if we've solved the problem
        if accuracy == 100.0 and loss.data < 0.01 and epoch > 100:
            print(f"XOR problem solved at epoch {epoch}!")
            break

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
        pred = model(x_point).data[0][0]
        Z.append(1 if pred >= 0.5 else 0)

    Z = np.array(Z).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.RdBu_r, edgecolors="k")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Decision Boundary")

    plt.tight_layout()
    plt.savefig("xor_solved.png")

    # Test the model
    print("\nPredictions vs. Actual:")
    for i in range(len(X)):
        input_data = X[i]
        actual = y[i][0]

        # Make prediction
        x_input = Tensor(input_data.reshape(1, -1))
        prediction = model(x_input).data[0][0]
        predicted_class = 1 if prediction >= 0.5 else 0

        print(
            f"Input: {input_data}, Predicted: {prediction:.4f} -> {predicted_class}, Actual: {actual}"
        )

    # Calculate accuracy
    outputs = model(X_tensor).data
    threshold = 0.5
    predicted_classes = (outputs >= threshold).astype(int)
    actual_classes = y.astype(int)
    accuracy = np.mean(predicted_classes == actual_classes) * 100

    print(f"\nAccuracy: {accuracy:.2f}%")

    return model, accuracy


if __name__ == "__main__":
    solve_xor()
