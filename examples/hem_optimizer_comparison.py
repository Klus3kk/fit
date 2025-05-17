"""
Fixed XOR problem comparison of different optimizers.

This version implements proper training settings for each optimizer,
including the High Error Margin (HEM) loss which significantly improves
convergence on the XOR problem.
"""

import matplotlib.pyplot as plt
import numpy as np

from core.tensor import Tensor
from nn.activations import Tanh
from nn.linear import Linear
from nn.sequential import Sequential
from train.loss import MSELoss, CrossEntropyLoss
from train.optim import SGD, Adam, SGDMomentum
from train.optim_lion import Lion
from train.hem_loss import HEMLoss


def train_xor_with_optimizer(optimizer_name, epochs=2000, verbose=True, use_hem=False):
    """
    Train a model to solve the XOR problem with a specific optimizer.

    Args:
        optimizer_name: Name of the optimizer to use
        epochs: Number of training epochs
        verbose: Whether to print progress
        use_hem: Whether to use HEM loss instead of MSE

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

    # Create model with architecture that can learn XOR
    hidden_size = 8  # Larger hidden size for better capacity

    model = Sequential(Linear(2, hidden_size), Tanh(), Linear(hidden_size, 1))

    # Different initializations for each optimizer
    # These are adjusted to help each optimizer converge
    scale = 0.5
    model.layers[0].weight.data = np.random.uniform(-scale, scale, (2, hidden_size))
    model.layers[0].bias.data = np.random.uniform(-0.1, 0.1, hidden_size)
    model.layers[2].weight.data = np.random.uniform(-scale, scale, (hidden_size, 1))
    model.layers[2].bias.data = np.random.uniform(-0.1, 0.1, 1)

    # Create loss function (HEM or MSE)
    if use_hem:
        loss_fn = HEMLoss(margin=0.5)
    else:
        loss_fn = MSELoss()

    # Create optimizer based on name, with appropriate learning rates
    if optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=0.1)
    elif optimizer_name == "SGDMomentum":
        optimizer = SGDMomentum(model.parameters(), lr=0.05, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=0.03)
    elif optimizer_name == "Lion":
        optimizer = Lion(model.parameters(), lr=0.05)
    elif optimizer_name == "HEM-Adam":
        optimizer = Adam(model.parameters(), lr=0.03)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Training loop
    losses = []
    accuracies = []

    for epoch in range(1, epochs + 1):
        # Forward pass
        outputs = model(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        losses.append(float(loss.data))

        # Compute accuracy
        with_threshold = lambda x: 1 if x >= 0.5 else 0
        predicted = np.array([with_threshold(x[0]) for x in outputs.data])
        actual = y.reshape(-1)
        accuracy = np.mean(predicted == actual) * 100
        accuracies.append(accuracy)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        # Print progress
        if verbose and epoch % 100 == 0:
            print(
                f"{optimizer_name} - Epoch {epoch}/{epochs}, Loss: {loss.data:.4f}, Accuracy: {accuracy:.1f}%"
            )

    # Final evaluation
    outputs = model(X_tensor).data
    with_threshold = lambda x: 1 if x >= 0.5 else 0
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
            print(
                f"Input: {input_data}, Predicted: {prediction:.4f} -> {predicted_class}, Actual: {actual}"
            )

    return {
        "losses": losses,
        "accuracies": accuracies,
        "final_accuracy": accuracy,
        "predictions": outputs.flatten(),
        "optimizer": optimizer_name,
    }


def compare_optimizers():
    """
    Compare different optimizers on the XOR problem.
    """
    # Set up plot
    plt.figure(figsize=(16, 12))

    # List of optimizers to compare
    optimizers = ["SGD", "SGDMomentum", "Adam", "Lion", "HEM-Adam"]
    results = {}

    # Train with each optimizer
    for optimizer_name in optimizers:
        print(f"\nTraining with {optimizer_name}...")
        # Use HEM loss for HEM-Adam
        use_hem = optimizer_name == "HEM-Adam"
        result = train_xor_with_optimizer(
            optimizer_name, epochs=2000, verbose=True, use_hem=use_hem
        )
        results[optimizer_name] = result

    # Plot loss curves
    plt.subplot(2, 2, 1)
    for optimizer_name, result in results.items():
        plt.plot(result["losses"], label=f"{optimizer_name}")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot accuracy curves
    plt.subplot(2, 2, 2)
    for optimizer_name, result in results.items():
        plt.plot(result["accuracies"], label=f"{optimizer_name}")
    plt.title("Training Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 105)

    # Plot accuracy comparison
    plt.subplot(2, 2, 3)
    accuracies = [results[opt]["final_accuracy"] for opt in optimizers]
    bars = plt.bar(optimizers, accuracies)
    plt.title("Final Accuracy Comparison")
    plt.xlabel("Optimizer")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)  # Max 100% with some margin

    # Add accuracy values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{accuracies[i]:.1f}%",
            ha="center",
            va="bottom",
        )

    # Plot decision boundaries
    plt.subplot(2, 2, 4)

    # Create a 2D meshgrid for decision boundary
    h = 0.01
    x_min, x_max = -0.2, 1.2
    y_min, y_max = -0.2, 1.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Create a colorful plot with decision boundaries for HEM and one other optimizer
    from matplotlib.colors import ListedColormap

    # Choose which optimizer to compare with HEM
    comparison_optimizer = "Adam"

    # Create datasets to store decision boundaries
    Z_hem = np.zeros(grid_points.shape[0])
    Z_other = np.zeros(grid_points.shape[0])

    # Create a model and retrain to get decision boundary
    # For HEM-Adam
    model = Sequential(Linear(2, 8), Tanh(), Linear(8, 1))
    use_hem = True
    result = train_xor_with_optimizer(
        "HEM-Adam", epochs=2000, verbose=False, use_hem=use_hem
    )

    # Get decision boundary for HEM
    for i, point in enumerate(grid_points):
        x_point = Tensor(point.reshape(1, -1))
        prediction = model(x_point).data[0][0]
        Z_hem[i] = 1 if prediction >= 0.5 else 0

    # For comparison optimizer
    model = Sequential(Linear(2, 8), Tanh(), Linear(8, 1))
    use_hem = False
    result = train_xor_with_optimizer(
        comparison_optimizer, epochs=2000, verbose=False, use_hem=use_hem
    )

    # Get decision boundary
    for i, point in enumerate(grid_points):
        x_point = Tensor(point.reshape(1, -1))
        prediction = model(x_point).data[0][0]
        Z_other[i] = 1 if prediction >= 0.5 else 0

    # Convert to 2D grids
    Z_hem = Z_hem.reshape(xx.shape)
    Z_other = Z_other.reshape(xx.shape)

    # Plot the decision boundaries
    plt.contourf(
        xx,
        yy,
        Z_hem,
        alpha=0.3,
        levels=[-0.5, 0.5, 1.5],
        colors=["#ff9999", "#99ccff"],
        label="HEM-Adam",
    )
    plt.contour(
        xx,
        yy,
        Z_other,
        colors=["red"],
        linestyles=["--"],
        levels=[0.5],
        label=comparison_optimizer,
    )

    # Plot the training points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors="k")

    plt.title(f"Decision Boundaries: HEM vs {comparison_optimizer}")
    plt.xlabel("X1")
    plt.ylabel("X2")

    # Add a custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="blue", lw=1, label="HEM-Adam"),
        Line2D([0], [0], color="red", linestyle="--", lw=1, label=comparison_optimizer),
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig("optimizer_comparison.png")

    # Generate a summary report
    print("\nOptimizer Performance Summary:")
    for opt in optimizers:
        print(
            f"- {opt}: {results[opt]['final_accuracy']:.1f}% accuracy, final loss = {results[opt]['losses'][-1]:.6f}"
        )

    print("\nComparison complete. See optimizer_comparison.png for results.")


if __name__ == "__main__":
    compare_optimizers()
