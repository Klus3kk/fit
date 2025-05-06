# Example usage in examples/mnist.py (to be implemented later)
import numpy as np
from core.tensor import Tensor
from nn.sequential import Sequential
from nn.linear import Linear
from nn.activations import ReLU, Softmax, Dropout
from nn.normalization import BatchNorm
from train.loss import CrossEntropyLoss
from train.optim import Adam
from train.scheduler import StepLR
from monitor.tracker import TrainingTracker
from utils.data import Dataset, DataLoader
from train.engine import train, evaluate
from nn.model_io import save_model, load_model


def main():
    # Load data (placeholder for MNIST)
    # In a real implementation, you'd load actual MNIST data
    train_data = np.random.randn(1000, 784)
    train_targets = np.random.randint(0, 10, 1000)

    val_data = np.random.randn(200, 784)
    val_targets = np.random.randint(0, 10, 200)

    # Create datasets
    train_dataset = Dataset(train_data, train_targets)
    val_dataset = Dataset(val_data, val_targets)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    model = Sequential(
        Linear(784, 128),
        BatchNorm(128),
        ReLU(),
        Dropout(0.3),
        Linear(128, 64),
        BatchNorm(64),
        ReLU(),
        Dropout(0.3),
        Linear(64, 10),
        Softmax()
    )

    # Create loss function and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Create learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Create tracker with early stopping
    tracker = TrainingTracker(
        experiment_name="mnist_example",
        early_stopping={"patience": 5, "metric": "val_loss", "min_delta": 0.001}
    )

    # Train model
    tracker = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=50,
        scheduler=scheduler,
        tracker=tracker
    )

    # Show final summary
    tracker.summary(show_best=True)

    # Plot training metrics
    tracker.plot(save_path="training_metrics.png")

    # Save model
    save_model(model, "mnist_model.pkl")

    # Load model (demonstration)
    loaded_model = load_model("mnist_model.pkl")

    # Evaluate loaded model
    test_metrics = evaluate(loaded_model, val_loader, loss_fn)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()