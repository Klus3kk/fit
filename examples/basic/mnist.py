import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from fit.core.tensor import Tensor
from fit.nn.modules.activation import ReLU, Softmax
from fit.nn.modules.linear import Linear
from fit.nn.utils.model_io import load_model, save_model
from fit.nn.modules.container import Sequential
from fit.loss.classification import CrossEntropyLoss
from fit.optim.adam import Adam
from fit.data.dataset import Dataset
from fit.data.dataloader import DataLoader


def load_mnist_data(path="./data/mnist"):
    """
    Load the MNIST dataset using scikit-learn's fetch_openml.

    Args:
        path: Directory to cache the MNIST data

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    # Create data directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    print("Fetching MNIST data from OpenML (this may take a moment)...")
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        parser="auto",
        cache=True,
        data_home=path,
        as_frame=False,
    )

    # Get features and targets
    X = mnist.data.astype("float32") / 255.0
    y = mnist.target.astype("int")

    # Split into train and test sets
    # MNIST has 60,000 training images and 10,000 test images
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print(
        f"MNIST dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples"
    )

    return X_train, y_train, X_test, y_test


def train_and_evaluate_mnist():
    """
    Train and evaluate a model on MNIST with careful parameter handling
    to ensure proper learning.
    """
    # Load data
    X_train, y_train, X_test, y_test = load_mnist_data()

    # Use a subset for faster training in this example
    subset_size = 5000
    train_indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]

    # Take smaller test set too
    test_subset_size = 1000
    test_indices = np.random.choice(len(X_test), test_subset_size, replace=False)
    X_test_subset = X_test[test_indices]
    y_test_subset = y_test[test_indices]

    print(
        f"Using {len(X_train_subset)} training samples and {len(X_test_subset)} test samples"
    )

    # Create model: 784 -> 128 -> 64 -> 10
    model = Sequential(
        Linear(784, 128), ReLU(), Linear(128, 64), ReLU(), Linear(64, 10), Softmax()
    )

    print("Model architecture:")
    print("784 (input) -> 128 -> ReLU -> 64 -> ReLU -> 10 -> Softmax (output)")

    # Create dataset and dataloader
    train_dataset = Dataset(X_train_subset, y_train_subset)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    print(f"\nStarting training with batch size 32...")

    # Training loop
    epochs = 10
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # Zero gradients
            for param in model.parameters():
                param.grad = None

            # Forward pass
            output = model(batch_x)
            loss = criterion(output, batch_y)

            # Backward pass
            try:
                loss.backward()
                optimizer.step()

                epoch_loss += loss.data
                batch_count += 1

                # Print progress every 20 batches
                if batch_idx % 20 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.data:.4f}"
                    )

            except Exception as e:
                print(f"Error in training: {e}")
                return False

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")

        # Evaluate on a small subset every few epochs
        if (epoch + 1) % 3 == 0:
            accuracy = evaluate_model(model, X_test_subset[:100], y_test_subset[:100])
            print(f"Test Accuracy (100 samples): {accuracy:.2%}")

    # Final evaluation
    print("\nFinal evaluation on test set...")
    final_accuracy = evaluate_model(model, X_test_subset, y_test_subset)
    print(f"Final Test Accuracy: {final_accuracy:.2%}")

    # Save the model
    model_path = "mnist_model.pkl"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    # Plot training loss
    try:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        # Show some test predictions
        plt.subplot(1, 2, 2)
        show_predictions(model, X_test_subset[:16], y_test_subset[:16])

        plt.tight_layout()
        plt.savefig("mnist_results.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("Results saved as 'mnist_results.png'")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    return final_accuracy > 0.7  # Return True if we got at least 70% accuracy


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Accuracy as a float
    """
    correct = 0
    total = len(X_test)

    for i in range(total):
        # Get single sample
        x = Tensor([X_test[i]], requires_grad=False)

        # Get prediction
        output = model(x)
        predicted = np.argmax(output.data)
        actual = y_test[i]

        if predicted == actual:
            correct += 1

    accuracy = correct / total
    return accuracy


def show_predictions(model, X_samples, y_true, num_samples=16):
    """
    Show model predictions on sample images.

    Args:
        model: Trained model
        X_samples: Sample images
        y_true: True labels
        num_samples: Number of samples to show
    """
    plt.figure(figsize=(12, 8))

    for i in range(min(num_samples, len(X_samples))):
        plt.subplot(4, 4, i + 1)

        # Reshape image for display
        image = X_samples[i].reshape(28, 28)
        plt.imshow(image, cmap="gray")

        # Get prediction
        x = Tensor([X_samples[i]], requires_grad=False)
        output = model(x)
        predicted = np.argmax(output.data)
        actual = y_true[i]

        # Set title with prediction
        color = "green" if predicted == actual else "red"
        plt.title(f"Pred: {predicted}, True: {actual}", color=color)
        plt.axis("off")

    plt.suptitle("MNIST Predictions (Green=Correct, Red=Wrong)")


def demo_saved_model():
    """
    Demonstrate loading a saved model and making predictions.
    """
    model_path = "mnist_model.pkl"

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train a model first.")
        return

    print("Loading saved model...")

    try:
        # Load the model
        model = load_model(model_path)
        print("Model loaded successfully!")

        # Load some test data
        _, _, X_test, y_test = load_mnist_data()

        # Test on a few samples
        print("\nTesting loaded model:")
        for i in range(5):
            x = Tensor([X_test[i]], requires_grad=False)
            output = model(x)
            predicted = np.argmax(output.data)
            confidence = np.max(output.data)
            actual = y_test[i]

            print(
                f"Sample {i}: Predicted {predicted} (confidence: {confidence:.3f}), Actual: {actual}"
            )

    except Exception as e:
        print(f"Error loading model: {e}")


if __name__ == "__main__":
    print("MNIST Classification Example")
    print("=" * 40)

    # Ask user what to do
    choice = input("Enter 1 to train new model, 2 to test saved model, or 3 for both: ")

    if choice in ["1", "3"]:
        print("\nTraining new model...")
        success = train_and_evaluate_mnist()

        if success:
            print("\n🎉 MNIST training completed successfully!")
        else:
            print("\n❌ MNIST training failed")

    if choice in ["2", "3"]:
        print("\nTesting saved model...")
        demo_saved_model()

    print("\nExample completed!")
