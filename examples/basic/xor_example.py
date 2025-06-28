"""
Fixed XOR example using proper batch training and loss function.
"""

import numpy as np
import matplotlib.pyplot as plt

from fit.core.tensor import Tensor
from fit.nn.modules.activation import Tanh, Sigmoid
from fit.nn.modules.linear import Linear
from fit.nn.modules.container import Sequential
from fit.optim.adam import Adam
from fit.loss.regression import MSELoss


def solve_xor():
    """
    Solve the XOR problem using proper batch training.
    """
    print("Solving XOR problem with neural network...")

    # XOR dataset - use proper batch format
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

    print("Dataset:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {y[i][0]}")

    # Convert to tensors - X needs gradients for proper backprop
    X_tensor = Tensor(X, requires_grad=True)
    y_tensor = Tensor(y, requires_grad=False)

    # Create network: 2 -> 8 -> 1 with proper activations
    # Using Sigmoid output so outputs are in [0,1] range like targets
    model = Sequential(
        Linear(2, 8),
        Tanh(),
        Linear(8, 1),
        Sigmoid()  # Output in [0,1] range
    )

    # Initialize weights to break symmetry
    # This is crucial for XOR convergence
    init_scale = 2.0
    with_grad = model.layers[0].weight.requires_grad
    model.layers[0].weight.requires_grad = False
    
    # Set alternating pattern in first layer
    for i in range(8):
        if i % 2 == 0:
            model.layers[0].weight.data[i, 0] = init_scale
            model.layers[0].weight.data[i, 1] = -init_scale
        else:
            model.layers[0].weight.data[i, 0] = -init_scale
            model.layers[0].weight.data[i, 1] = init_scale
    
    # Set alternating bias
    for i in range(8):
        if i < 4:
            model.layers[0].bias.data[i] = 0.5
        else:
            model.layers[0].bias.data[i] = -0.5
    
    model.layers[0].weight.requires_grad = with_grad

    # Use proper MSE loss function
    loss_fn = MSELoss()
    
    # Create optimizer with higher learning rate for XOR
    optimizer = Adam(model.parameters(), lr=0.01)

    # Training loop - batch training
    epochs = 1000
    losses = []

    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass on entire batch
        output = model(X_tensor)

        # Calculate loss
        loss = loss_fn(output, y_tensor)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Record loss
        losses.append(loss.data)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data:.6f}")

    # Test the trained model
    print("\nTesting trained model:")
    print("Input -> Output (Target)")
    
    test_output = model(X_tensor)
    for i in range(len(X)):
        predicted = test_output.data[i][0]
        actual = y[i][0]
        print(f"{X[i]} -> {predicted:.4f} ({actual})")

    # Check if learning was successful
    print("\nEvaluating results:")
    correct = 0
    for i in range(len(X)):
        predicted = 1 if test_output.data[i][0] > 0.5 else 0
        actual = int(y[i][0])
        if predicted == actual:
            correct += 1
        print(
            f"Input: {X[i]}, Predicted: {predicted}, Actual: {actual}, {'✓' if predicted == actual else '✗'}"
        )

    accuracy = correct / len(X)
    print(f"\nAccuracy: {accuracy:.2%}")

    # Plot results
    try:
        plt.figure(figsize=(12, 5))

        # Plot loss curve
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.yscale('log')  # Log scale to see convergence better

        # Plot decision boundary
        plt.subplot(1, 2, 2)
        
        # Create a grid to visualize the decision boundary
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Get model predictions for the grid
        grid_tensor = Tensor(grid_points, requires_grad=False)
        Z = model(grid_tensor).data.reshape(xx.shape)

        # Plot decision boundary
        plt.contourf(xx, yy, Z, levels=20, alpha=0.8, cmap="RdYlBu")
        plt.colorbar(label="Output")

        # Plot data points
        colors = ["red", "blue"]
        for i in range(len(X)):
            plt.scatter(
                X[i][0], X[i][1], c=colors[int(y[i][0])], s=200, 
                edgecolors="black", linewidth=2
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

    return accuracy > 0.75


if __name__ == "__main__":
    success = solve_xor()
    if success:
        print("\nXOR problem solved successfully!")
    else:
        print("\nFailed to solve XOR problem")