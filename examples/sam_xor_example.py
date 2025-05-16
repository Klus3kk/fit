"""
Updated SAM optimizer example for the XOR problem.
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


def run_sam_xor_example():
    # Create a simple dataset - XOR problem
    X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), requires_grad=False)
    y = Tensor(np.array([0, 1, 1, 0]), requires_grad=False)

    # Create a model with adequate capacity
    model = Sequential(
        Linear(2, 32),
        BatchNorm(32),
        ReLU(),
        Linear(32, 16),
        BatchNorm(16),
        ReLU(),
        Linear(16, 2),
        Softmax(),
    )

    # Print model summary
    model.summary((2,))

    # Create loss function
    loss_fn = CrossEntropyLoss()

    # Create optimizers
    base_optimizer = Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.1)

    # Create a tracker to monitor training
    tracker = TrainingTracker(experiment_name="sam_xor")

    # Loss closure for SAM optimizer
    def loss_closure():
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        return loss

    # Train for 300 epochs
    print("\nTraining with SAM optimizer...")
    best_accuracy = 0

    for epoch in range(1, 301):
        # Start a new epoch in the tracker
        tracker.start_epoch()

        # First SAM step: compute gradients, perturb weights
        loss = optimizer.first_step(loss_closure)

        # Second SAM step: compute gradients at perturbed weights,
        # restore original weights, then update with base optimizer
        optimizer.second_step(loss_closure)

        # Compute accuracy
        outputs = model(X)
        predicted = np.argmax(outputs.data, axis=1)
        accuracy = np.mean(predicted == y.data)

        # Update best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        # Add learning rate decay after 150 epochs
        if epoch == 150:
            base_optimizer.lr *= 0.5
            print(f"Reducing learning rate to {base_optimizer.lr}")

        # Log metrics
        tracker.log(loss=loss.data, acc=accuracy, lr=base_optimizer.lr)

        # Print progress
        if epoch % 20 == 0 or accuracy == 1.0:
            print(
                f"Epoch {epoch}/{300}: Loss = {loss.data:.4f}, Accuracy = {accuracy:.2f}"
            )

        # Early stopping
        if accuracy == 1.0 and epoch > 50:
            print(f"Reached 100% accuracy at epoch {epoch}. Early stopping.")
            break

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

    # Save and plot the training metrics
    tracker.export(format="json")
    tracker.plot(save_path="sam_xor_plot.png")

    print("\nSAM training completed!")


if __name__ == "__main__":
    run_sam_xor_example()
