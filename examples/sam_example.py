"""
Example usage of the SAM optimizer with FIT framework.

This example shows how to use the Sharpness-Aware Minimization optimizer
for improved model generalization on the XOR problem.
"""

import numpy as np

from core.tensor import Tensor
from monitor.tracker import TrainingTracker
from nn.activations import ReLU, Softmax
from nn.linear import Linear
from nn.normalization import BatchNorm
from nn.sequential import Sequential
from train.loss import CrossEntropyLoss
from train.optim import Adam, SGD
from train.optim_sam import SAM


def run_sam_example():
    # Create a simple dataset - XOR problem
    X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), requires_grad=False)
    y = Tensor(np.array([0, 1, 1, 0]), requires_grad=False)

    # Create a small model
    model = Sequential(
        Linear(2, 32),  # Larger first layer (was 16)
        BatchNorm(32),
        ReLU(),
        Linear(32, 16),  # add middle layer for more expressivity
        BatchNorm(16),
        ReLU(),
        Linear(16, 2),
        Softmax(),
    )

    # Print model summary
    model.summary((2,))

    # Create loss function
    loss_fn = CrossEntropyLoss()

    # We'll compare Adam optimizer vs. SAM with Adam as base optimizer
    # First, define the base Adam optimizer
    base_optimizer = Adam(model.parameters(), lr=0.02, weight_decay=1e-4)

    # Create SAM optimizer with Adam as the base
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.1)

    # Create a tracker to monitor training
    tracker = TrainingTracker(experiment_name="xor_with_sam")

    # We need a closure to re-compute the loss
    def loss_closure():
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        return loss

    # Train for 200 epochs
    print("\nTraining with SAM optimizer...")
    for epoch in range(1, 201):
        # Start a new epoch in the tracker
        tracker.start_epoch()

        # First SAM step: compute gradients, perturb weights
        loss = optimizer.first_step(loss_closure)

        # Second SAM step: compute gradients at perturbed weights,
        # restore original weights, then update with base optimizer
        optimizer.second_step(loss_closure)

        # Forward pass to compute accuracy
        outputs = model(X)
        predicted = np.argmax(outputs.data, axis=1)
        accuracy = np.mean(predicted == y.data)

        # Log metrics
        tracker.log(loss=loss.data, acc=accuracy)

        # Print progress occasionally
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}/{200}: Loss = {loss.data:.4f}, Accuracy = {accuracy:.2f}"
            )

    # Display final model performance
    outputs = model(X)
    predicted = np.argmax(outputs.data, axis=1)
    print("\nFinal Predictions:")
    print(outputs.data)
    print("\nPredicted classes:", predicted)
    print("Actual classes:   ", y.data)
    print(f"Final Accuracy: {np.mean(predicted == y.data) * 100:.2f}%")

    # Show training summary
    tracker.summary(style="box")

    # Optional: save and plot the training metrics
    tracker.export(format="json")
    tracker.plot(save_path="sam_training_plot.png")

    print("\nSAM training completed!")


if __name__ == "__main__":
    run_sam_example()
