"""
Demonstration of the autograd system on the XOR problem.

This example shows how to use the autograd system to implement and train
a simple neural network to solve the XOR problem.
"""

import numpy as np
import matplotlib.pyplot as plt

from core.tensor import Tensor
import core.ops as ops


def plot_decision_boundary(
    model, x_min=-0.1, x_max=1.1, y_min=-0.1, y_max=1.1, mesh_size=0.01
):
    """Plot the decision boundary of a model."""
    # Create mesh grid
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_size), np.arange(y_min, y_max, mesh_size)
    )

    # Create input features
    X_mesh = np.c_[xx.ravel(), yy.ravel()]

    # Predict outputs
    Z = []
    for x in X_mesh:
        # Forward pass
        h1 = ops.matmul(Tensor([x]), W1) + b1
        h1_act = ops.tanh(h1)
        logits = ops.matmul(h1_act, W2) + b2
        pred = ops.sigmoid(logits)
        Z.append(pred.data[0, 0])

    # Reshape predictions to match mesh grid
    Z = np.array(Z).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors="k")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("XOR Decision Boundary")


# Define XOR problem data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Convert to tensors
X_tensor = Tensor(X)
y_tensor = Tensor(y)

# Define network parameters
np.random.seed(42)  # For reproducibility
W1 = Tensor(np.random.randn(2, 4) * 0.1, requires_grad=True)
b1 = Tensor(np.zeros(4), requires_grad=True)
W2 = Tensor(np.random.randn(4, 1) * 0.1, requires_grad=True)
b2 = Tensor(np.zeros(1), requires_grad=True)

# Training parameters
learning_rate = 0.1
num_epochs = 1000
losses = []

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    h1 = ops.matmul(X_tensor, W1) + b1
    h1_act = ops.tanh(h1)
    logits = ops.matmul(h1_act, W2) + b2
    outputs = ops.sigmoid(logits)

    # Compute loss (Mean Squared Error)
    loss = ops.mse_loss(outputs, y_tensor)
    losses.append(loss.data)

    # Backward pass
    loss.backward()

    # Update parameters using gradient descent
    W1.data -= learning_rate * W1.grad
    b1.data -= learning_rate * b1.grad
    W2.data -= learning_rate * W2.grad
    b2.data -= learning_rate * b2.grad

    # Zero gradients for next iteration
    W1.zero_grad()
    b1.zero_grad()
    W2.zero_grad()
    b2.zero_grad()

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.data:.4f}")

# Make predictions
h1 = ops.matmul(X_tensor, W1) + b1
h1_act = ops.tanh(h1)
logits = ops.matmul(h1_act, W2) + b2
predictions = ops.sigmoid(logits)

# Convert predictions to binary
binary_predictions = (predictions.data > 0.5).astype(int)

# Print results
print("\nPredictions vs. Actual:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted: {binary_predictions[i][0]}, Actual: {y[i][0]}")

# Plot loss curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

# Plot decision boundary
plt.subplot(1, 2, 2)
plot_decision_boundary(None)  # Model is defined through global variables
plt.tight_layout()
plt.savefig("xor_results.png")
plt.show()

print(
    "\nTraining complete! Decision boundary and loss curve saved to 'xor_results.png'"
)
