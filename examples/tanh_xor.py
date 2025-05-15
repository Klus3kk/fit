"""
Basic XOR problem solver using Tanh activation.
"""

import numpy as np

from core.tensor import Tensor
from nn.activations import Tanh
from nn.linear import Linear
from nn.sequential import Sequential
from train.loss import MSELoss
from train.optim import SGD


def solve_xor():
    """Train a simple neural network to solve the XOR problem."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Convert to tensors
    X_tensor = Tensor(X, requires_grad=True)
    y_tensor = Tensor(y, requires_grad=False)

    # Create model with proper initialization for XOR
    model = Sequential(
        Linear(2, 4),
        Tanh(),  # Tanh works better than ReLU for XOR
        Linear(4, 1)
    )

    # Initialize weights to break symmetry
    model.layers[0].weight.data = np.array([
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0]
    ])
    model.layers[0].bias.data = np.array([0.0, 0.0, 1.0, 1.0])

    # Create loss function and optimizer
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    # Training loop
    epochs = 1000
    for epoch in range(1, epochs + 1):
        # Forward pass
        outputs = model(X_tensor)
        loss = loss_fn(outputs, y_tensor)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.data:.4f}")

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

        print(f"Input: {input_data}, Predicted: {prediction:.4f} -> {predicted_class}, Actual: {actual}")

    # Calculate accuracy
    outputs = model(X_tensor).data
    predicted_classes = np.array([with_threshold(x[0]) for x in outputs])
    actual_classes = y.flatten()
    accuracy = np.mean(predicted_classes == actual_classes) * 100

    print(f"\nAccuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    solve_xor()
