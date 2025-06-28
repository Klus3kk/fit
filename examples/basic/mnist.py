import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from fit.core.tensor import Tensor
from fit.nn.modules.activation import ReLU
from fit.nn.modules.linear import Linear
from fit.nn.modules.container import Sequential
from fit.loss.classification import CrossEntropyLoss
from fit.optim.adam import Adam
from fit.data.dataset import Dataset
from fit.data.dataloader import DataLoader


def load_mnist_data(path="./data/mnist"):
    """Load the MNIST dataset using scikit-learn's fetch_openml."""
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
    X = mnist.data.astype("float32") / 255.0  # Normalize to [0,1]
    y = mnist.target.astype("int")
    
    # Split into train and test sets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    print(f"MNIST dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples")
    return X_train, y_train, X_test, y_test


def train_and_evaluate_mnist():
    """Train and evaluate a model on MNIST with proper training loop."""
    
    # Load data
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Use subset for faster training
    subset_size = 5000
    train_indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]
    
    test_subset_size = 1000
    test_indices = np.random.choice(len(X_test), test_subset_size, replace=False)
    X_test_subset = X_test[test_indices]
    y_test_subset = y_test[test_indices]
    
    print(f"Using {len(X_train_subset)} training samples and {len(X_test_subset)} test samples")
    
    # Create model: 784 -> 256 -> 128 -> 10 (NO SOFTMAX!)
    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 128), 
        ReLU(),
        Linear(128, 10)  # Raw logits - CrossEntropyLoss will handle softmax
    )
    
    print("Model architecture:")
    print("784 (input) -> 256 -> ReLU -> 128 -> ReLU -> 10 (logits)")
    
    # Debug: Check model parameters
    print(f"\nModel parameters:")
    for i, param in enumerate(model.parameters()):
        print(f"  Param {i}: shape {param.data.shape}, requires_grad={param.requires_grad}")
    
    # Create dataset and dataloader
    train_dataset = Dataset(X_train_subset, y_train_subset)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)  # Standard Adam LR
    
    print(f"\nStarting training with batch size 32...")
    
    # Training loop
    epochs = 10
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        correct = 0
        total = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # CORRECT TRAINING ORDER:
            # 1. Zero gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            output = model(batch_x)
            loss = criterion(output, batch_y)
            
            # 3. Backward pass
            loss.backward()
        
            # print("After backward:")
            # for i, param in enumerate(model.parameters()):
            #     if param.grad is not None:
            #         print(f"  Param {i}: HAS gradient, shape {param.grad.shape}")
            #     else:
            #         print(f"  Param {i}: NO gradient! (shape {param.data.shape})")
            # 4. Update parameters
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.data
            batch_count += 1
            
            # Calculate accuracy
            predictions = np.argmax(output.data, axis=1)
            targets = batch_y.data
            correct += np.sum(predictions == targets)
            total += len(targets)
            
            # Print progress every 20 batches
            if batch_idx % 20 == 0:
                current_acc = (correct / total) * 100 if total > 0 else 0
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.data:.4f}, Acc: {current_acc:.1f}%")
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / batch_count
        epoch_accuracy = (correct / total) * 100
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
        # Test every 3 epochs
        if (epoch + 1) % 3 == 0:
            test_accuracy = evaluate_model(model, X_test_subset[:100], y_test_subset[:100])
            print(f"Test Accuracy (100 samples): {test_accuracy:.2f}%")
    
    # Final evaluation
    print("\nFinal evaluation on test set...")
    final_test_accuracy = evaluate_model(model, X_test_subset, y_test_subset)
    print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")
    
    sample_images = X_test_subset[:16].reshape(-1, 28, 28)
    sample_labels = y_test_subset[:16]
    
    # Get predictions
    sample_tensor = Tensor(X_test_subset[:16])
    with_no_grad = True  # Disable gradient computation
    sample_output = model(sample_tensor)
    sample_predictions = np.argmax(sample_output.data, axis=1)
    
    # Plot images with predictions
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(sample_images[i], cmap='gray')
        color = 'green' if sample_predictions[i] == sample_labels[i] else 'red'
        plt.title(f'Pred: {sample_predictions[i]}, True: {sample_labels[i]}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_results.png', dpi=150)
    print("Results saved as 'mnist_results.png'")
    
    # Success criteria
    if final_test_accuracy > 85.0:
        print("MNIST training successful!")
        return True
    else:
        print("MNIST training failed")
        return False


def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy on test data."""
    test_tensor = Tensor(X_test)
    test_output = model(test_tensor)
    test_predictions = np.argmax(test_output.data, axis=1)
    accuracy = np.mean(test_predictions == y_test) * 100
    return accuracy


def main():
    """Main function."""
    print("=" * 40)
    
    print("Training new model...")
    success = train_and_evaluate_mnist()
        
    if not success:
        print("Training failed. Check your implementation.")
        return
    
    print("Example completed!")


if __name__ == "__main__":
    main()