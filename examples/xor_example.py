import numpy as np

from core.tensor import Tensor
from monitor.tracker import TrainingTracker
from nn.activations import Dropout, ReLU, Softmax
from nn.linear import Linear
from nn.normalization import BatchNorm
from nn.sequential import Sequential
from train.loss import CrossEntropyLoss
from train.optim import Adam
from train.scheduler import StepLR
from train.trainer import Trainer


def run_xor_example():
    # Create a simple dataset - XOR problem
    X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), requires_grad=False)
    y = Tensor(np.array([0, 1, 1, 0]), requires_grad=False)

    # Create a model with regularization techniques
    model = Sequential(
        Linear(2, 16),
        BatchNorm(16),  # Add batch normalization
        ReLU(),
        Dropout(0.2),  # Add dropout with lower rate
        Linear(16, 16),
        BatchNorm(16),  # Add batch normalization
        ReLU(),
        Dropout(0.2),  # Add dropout with lower rate
        Linear(16, 2),
        Softmax(),
    )

    # Print model summary
    model.summary((2,))

    # Create loss function, optimizer and tracker
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)  # Use Adam with weight decay
    # Reduce learning rate every 100 epochs
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    tracker = TrainingTracker()

    # Create trainer with gradient clipping
    trainer = Trainer(
        model,
        loss_fn,
        optimizer,
        tracker=tracker,
        scheduler=scheduler,
        grad_clip=1.0,  # Set maximum gradient norm
    )

    # Train the model
    trainer.fit(X, y, epochs=300, verbose=True)

    # Evaluate the model
    test_loss, test_acc = trainer.evaluate(X, y)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")

    # Forward pass to check predictions
    predictions = model(X)
    print("Predictions:")
    print(predictions.data)
    print("Predicted classes:")
    print(np.argmax(predictions.data, axis=1))
    print("Actual classes:")
    print(y.data)


if __name__ == "__main__":
    run_xor_example()
