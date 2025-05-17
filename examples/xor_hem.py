"""
Solving the XOR problem using the High Error Margin (HEM) loss.

This example demonstrates how HEM loss can effectively train neural networks
on the XOR problem, which is known to be challenging for traditional
optimization approaches.
"""

import numpy as np
import matplotlib.pyplot as plt

from core.tensor import Tensor
from nn.activations import Tanh
from nn.linear import Linear
from nn.sequential import Sequential
from train.optim import Adam

# Import the HEM loss
from train.hem_loss import HEMLoss


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
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)

    # Plot training points
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.RdBu_r, edgecolors="k")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.savefig("xor_hem_solution.png")
    plt.show()


# XOR problem data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Convert to tensors
X_tensor = Tensor(X, requires_grad=True)
y_tensor = Tensor(y, requires_grad=False)

# Create a model with adequate capacity for XOR
# Using Tanh activation which works well for this problem
model = Sequential(Linear(2, 8), Tanh(), Linear(8, 1))  # Larger hidden layer

# Initialize weights to break symmetry
# This helps overcome the "stuck at saddle point" problem
np.random.seed(42)
model.layers[0].weight.data = np.random.uniform(-0.5, 0.5, (2, 8))
model.layers[0].bias.data = np.random.uniform(-0.1, 0.1, 8)

# Create loss function using HEM loss
loss_fn = HEMLoss(margin=0.5)  # Margin parameter from the paper

# Create optimizer
optimizer = Adam(
    model.parameters(), lr=0.03
)  # Lower learning rate works better with HEM

# Training loop
epochs = 1000
losses = []

print("Training model with HEM loss...")
for epoch in range(1, epochs + 1):
    # Forward pass
    outputs = model(X_tensor)
    loss = loss_fn(outputs, y_tensor)
    losses.append(loss.data)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()
    optimizer.zero_grad()

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.data:.4f}")

# Plot training loss
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("xor_hem_loss.png")
plt.show()

# Test the model and calculate accuracy
threshold = 0.5  # Classification threshold
predictions = []
accuracies = []

for i, x in enumerate(X):
    # Convert input to tensor
    x_input = Tensor(x.reshape(1, -1))
    # Forward pass
    output = model(x_input)
    # Get prediction
    pred = 1 if output.data[0][0] >= threshold else 0
    predictions.append(pred)
    # Calculate accuracy for this sample
    accuracy = 1 if pred == y[i][0] else 0
    accuracies.append(accuracy)

# Print final results
print("\nFinal Results:")
print("XOR Problem Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i]}")

# Calculate overall accuracy
accuracy = sum(accuracies) / len(accuracies) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# Plot decision boundary
plot_decision_boundary(model, title="XOR Problem Solution with HEM Loss")

print("Training visualization saved to 'xor_hem_loss.png'")
print("Decision boundary saved to 'xor_hem_solution.png'")
