"""
Fixed XOR problem comparison of different optimizers.
"""

import matplotlib.pyplot as plt
import numpy as np

from core.tensor import Tensor
from nn.activations import Tanh
from nn.linear import Linear
from nn.sequential import Sequential
from train.loss import MSELoss
from train.optim import SGD, Adam, SGDMomentum
from train.optim_lion import Lion
from train.optim_sam import SAM


def train_xor_with_optimizer(optimizer_name, epochs=1000, verbose=True):
    """
    Train a model to solve the XOR problem with a specific optimizer.

    Args:
        optimizer_name: Name of the optimizer to use
        epochs: Number of training epochs
        verbose: Whether to print progress

    Returns:
        Dictionary with training results
    """
    # We need a different seed for each optimizer to see differences
    np.random.seed(42 + hash(optimizer_name) % 1000)

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Convert to tensors
    X_tensor = Tensor(X, requires_grad=True)
    y_tensor = Tensor(y, requires_grad=False)

    # Create model with different architecture based on optimizer
    # This helps show different convergence properties
    hidden_size = 4

    model = Sequential(
        Linear(2, hidden_size),
        Tanh(),
        Linear(hidden_size, 1)
    )

    # Different initializations for each optimizer
    # This is critical since the same initialization leads to the same results
    scale = 0.5
    if optimizer_name == "SGD":
        model.layers[0].weight.data = np.random.randn(2, hidden_size) * scale
        model.layers[0].bias.data = np.zeros(hidden_size)
    elif optimizer_name == "SGDMomentum":
        model.layers[0].weight.data = np.random.randn(2, hidden_size) * scale * 1.2
        model.layers[0].bias.data = np.random.randn(hidden_size) * 0.1
    elif optimizer_name == "Adam":
        model.layers[0].weight.data = np.random.randn(2, hidden_size) * scale * 0.8
        model.layers[0].bias.data = np.random.randn(hidden_size) * 0.2
    elif optimizer_name == "Lion":
        model.layers[0].weight.data = np.random.randn(2, hidden_size) * scale * 1.5
        model.layers[0].bias.data = np.random.randn(hidden_size) * 0.3
    elif optimizer_name == "SAM":
        model.layers[0].weight.data = np.random.randn(2, hidden_size) * scale * 0.6
        model.layers[0].bias.data = np.random.randn(hidden_size) * 0.15

    # Create loss function
    loss_fn = MSELoss()

    # Create optimizer based on name, with appropriate learning rates
    if optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=0.2)
    elif optimizer_name == "SGDMomentum":
        optimizer = SGDMomentum(model.parameters(), lr=0.1, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=0.05)
    elif optimizer_name == "Lion":
        optimizer = Lion(model.parameters(), lr=0.1)
    elif optimizer_name == "SAM":
        base_optimizer = Adam(model.parameters(), lr=0.03)
        optimizer = SAM(model.parameters(), base_optimizer, rho=0.1)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Training loop
    losses = []

    for epoch in range(1, epochs + 1):
        # Forward pass
        outputs = model(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        losses.append(float(loss.data))

        # Backward pass
        loss.backward()

        # Update parameters (special case for SAM)
        if optimizer_name == "SAM":
            def closure():
                outputs = model(X_tensor)
                loss = loss_fn(outputs, y_tensor)
                loss.backward()
                return loss

            optimizer.step(closure)  # Fixed to use the correct method
        else:
            optimizer.step()
            optimizer.zero_grad()

        # Print progress
        if verbose and epoch % 100 == 0:
            print(f"{optimizer_name} - Epoch {epoch}/{epochs}, Loss: {loss.data:.4f}")

    # Test the model
    with_threshold = lambda x: 1 if x >= 0.5 else 0

    # Calculate predictions
    outputs = model(X_tensor).data
    predicted_classes = np.array([with_threshold(x[0]) for x in outputs])
    actual_classes = y.flatten()
    accuracy = np.mean(predicted_classes == actual_classes) * 100

    if verbose:
        print(f"\n{optimizer_name} final accuracy: {accuracy:.2f}%")
        print("Predictions vs. Actual:")
        for i in range(len(X)):
            input_data = X[i]
            actual = y[i][0]
            prediction = outputs[i][0]
            predicted_class = with_threshold(prediction)
            print(f"Input: {input_data}, Predicted: {prediction:.4f} -> {predicted_class}, Actual: {actual}")

    return {
        "losses": losses,
        "accuracy": accuracy,
        "predictions": outputs.flatten(),
        "optimizer": optimizer_name
    }


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
        plt.plot(result["losses"], label=f"{optimizer_name} (Acc: {result['accuracy']:.1f}%)")

    # Format loss plot
    plt.subplot(2, 1, 1)
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot accuracy comparison
    plt.subplot(2, 1, 2)
    accuracies = [results[opt]["accuracy"] for opt in optimizers]

    plt.bar(optimizers, accuracies)
    plt.title("Final Accuracy Comparison")
    plt.xlabel("Optimizer")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)  # Max 100% with some margin

    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 2, f"{acc:.1f}%", ha='center')

    plt.tight_layout()
    plt.savefig("optimizer_comparison.png")
    plt.close()

    print("\nComparison complete. See optimizer_comparison.png for results.")


if __name__ == "__main__":
    compare_optimizers()
