import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from core.tensor import Tensor
from nn.activations import ReLU, Softmax
from nn.linear import Linear
from nn.model_io import load_model, save_model
from nn.sequential import Sequential
from train.loss import CrossEntropyLoss
from train.optim import Adam
from utils.data import DataLoader, Dataset


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
        "mnist_784", version=1, parser="auto", cache=True, data_home=path, as_frame=False
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
    print("Loading MNIST dataset...")

    # Load MNIST data
    train_images, train_labels, test_images, test_labels = load_mnist_data()

    # Use a small subset for faster training
    subset_size = 10000
    val_size = 2000

    # Create random indices for train/val split
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(len(train_images))
    train_idx, val_idx = indices[val_size:subset_size+val_size], indices[:val_size]

    # Create train/val splits
    train_data = train_images[train_idx]
    train_targets = train_labels[train_idx]
    val_data = train_images[val_idx]
    val_targets = train_labels[val_idx]

    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

    # Create datasets
    train_dataset = Dataset(train_data, train_targets)
    val_dataset = Dataset(val_data, val_targets)

    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create a simple model with proper initialization
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
    )

    # Use proper initialization for the layers
    # Xavier/Glorot initialization for the first layer
    w1_scale = np.sqrt(2.0 / (784 + 128))
    model.layers[0].weight.data = np.random.randn(784, 128) * w1_scale
    model.layers[0].bias.data = np.zeros(128)

    # Xavier/Glorot initialization for the second layer
    w2_scale = np.sqrt(2.0 / (128 + 10))
    model.layers[2].weight.data = np.random.randn(128, 10) * w2_scale
    model.layers[2].bias.data = np.zeros(10)

    # Print model summary
    print("\nModel architecture:")
    model.summary((784,))

    # Create loss function and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)  # Lower learning rate

    # Training parameters
    epochs = 15
    
    # Track metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Gradient clipping threshold
    grad_clip_threshold = 1.0

    # Training loop with gradient clipping
    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            # Forward pass
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping
            for param in model.parameters():
                if param.grad is not None:
                    # Calculate gradient norm
                    grad_norm = np.sqrt(np.sum(param.grad * param.grad))
                    if grad_norm > grad_clip_threshold:
                        param.grad = param.grad * (grad_clip_threshold / grad_norm)
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            # Update metrics
            batch_size = x.data.shape[0]
            total_loss += float(loss.data) * batch_size
            
            # For accuracy calculation
            predictions = np.argmax(outputs.data, axis=1)
            y_indices = y.data.astype(np.int32) if y.data.ndim == 1 else np.argmax(y.data, axis=1)
            batch_correct = np.sum(predictions == y_indices)
            
            correct += batch_correct
            total += batch_size
        
        # Calculate epoch metrics
        train_loss = total_loss / total
        train_accuracy = correct / total
        
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_total_loss = 0
        val_correct = 0
        val_total = 0
        
        for x, y in val_loader:
            # Forward pass (no backward needed)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            # Update metrics
            batch_size = x.data.shape[0]
            val_total_loss += float(loss.data) * batch_size
            
            # For accuracy calculation
            predictions = np.argmax(outputs.data, axis=1)
            y_indices = y.data.astype(np.int32) if y.data.ndim == 1 else np.argmax(y.data, axis=1)
            batch_correct = np.sum(predictions == y_indices)
            
            val_correct += batch_correct
            val_total += batch_size
        
        # Calculate validation metrics
        val_loss = val_total_loss / val_total
        val_accuracy = val_correct / val_total
        
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        
        # Print progress
        print(f"Epoch {epoch}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%")
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mnist_training_metrics.png')
    print("\nTraining plot saved to mnist_training_metrics.png")
    
    # Save model
    model_path = "mnist_model.pkl"
    save_model(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Evaluate on test set
    test_subset_size = 1000
    test_indices = np.random.choice(len(test_images), test_subset_size, replace=False)
    test_data = test_images[test_indices]
    test_labels = test_labels[test_indices]
    
    test_dataset = Dataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate test accuracy
    model.eval()
    test_correct = 0
    test_total = 0
    
    for x, y in test_loader:
        outputs = model(x)
        predictions = np.argmax(outputs.data, axis=1)
        y_indices = y.data.astype(np.int32) if y.data.ndim == 1 else np.argmax(y.data, axis=1)
        batch_correct = np.sum(predictions == y_indices)
        
        test_correct += batch_correct
        test_total += len(predictions)
    
    test_accuracy = test_correct / test_total
    print(f"\nTest accuracy: {test_accuracy*100:.2f}%")
    
    # Calculate per-class accuracy
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    
    for x, y in test_loader:
        outputs = model(x)
        predictions = np.argmax(outputs.data, axis=1)
        y_indices = y.data.astype(np.int32) if y.data.ndim == 1 else np.argmax(y.data, axis=1)
        
        for i, label in enumerate(y_indices):
            class_total[label] += 1
            if predictions[i] == label:
                class_correct[label] += 1
    
    print("\nPer-class accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            print(f"Digit {i}: {100 * class_correct[i] / class_total[i]:.2f}% accuracy")
    
    print("\nMNIST example completed successfully!")


if __name__ == "__main__":
    train_and_evaluate_mnist()