"""
This example demonstrates how to use the Sharpness-Aware Minimization (SAM) optimizer
for improved generalization and robustness on image classification tasks.

We'll compare standard optimizers against SAM on the CIFAR-10 dataset
to show the generalization benefits.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.tensor import Tensor
from nn.activations import ReLU
from nn.linear import Linear
from nn.sequential import Sequential
from nn.normalization import BatchNorm
from train.loss import CrossEntropyLoss
from train.optim import SGD, Adam
from train.optim_sam import SAM
from utils.data import DataLoader, Dataset


def load_cifar10_subset(n_samples=10000):
    """
    Load a subset of CIFAR-10 data for faster experimentation.
    
    Args:
        n_samples: Number of samples to load
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("Loading CIFAR-10 dataset...")
    
    # Load data from OpenML
    cifar = fetch_openml(name='CIFAR_10', version=1, parser='auto', as_frame=False)
    X = cifar.data.astype('float32') / 255.0
    y = cifar.target.astype('int')
    
    # Convert features to grayscale to simplify
    # Shape from (n, 3072) to (n, 1024) by averaging RGB channels
    X_reshaped = X.reshape(-1, 3, 32, 32)
    X_gray = X_reshaped.mean(axis=1).reshape(-1, 1024)
    
    # Take subset for faster training
    indices = np.random.permutation(len(X_gray))[:n_samples]
    X_subset = X_gray[indices]
    y_subset = y[indices]
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_subset, test_size=0.3, random_state=42, stratify=y_subset
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Dataset ready: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model():
    """
    Create a simple CNN-like model for CIFAR-10 classification.
    
    Returns:
        The model
    """
    # More complex architecture for CIFAR-10
    model = Sequential(
        Linear(1024, 512),
        BatchNorm(512),
        ReLU(),
        Linear(512, 256),
        BatchNorm(256),
        ReLU(),
        Linear(256, 10)
    )
    
    # Xavier/Glorot initialization for better training dynamics
    for i in [0, 3, 6]:  # Linear layers
        fan_in = model.layers[i].weight.data.shape[0]
        fan_out = model.layers[i].weight.data.shape[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        model.layers[i].weight.data = np.random.randn(*model.layers[i].weight.data.shape) * std
        model.layers[i].bias.data = np.zeros_like(model.layers[i].bias.data)
    
    return model


def train_and_evaluate(model, train_loader, val_loader, optimizer_name, learning_rate=0.01, epochs=10):
    """
    Train and evaluate a model using the specified optimizer.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer_name: Name of the optimizer to use ('sgd', 'adam', or 'sam')
        learning_rate: Learning rate
        epochs: Number of training epochs
        
    Returns:
        Dictionary with training history
    """
    # Create loss function
    loss_fn = CrossEntropyLoss()
    
    # Create optimizer
    if optimizer_name.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sam':
        # For SAM, we use Adam as the base optimizer
        base_optimizer = Adam(model.parameters(), lr=learning_rate)
        optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, adaptive=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_times': []
    }
    
    # Training loop
    print(f"\nTraining with {optimizer_name}...")
    
    for epoch in range(1, epochs + 1):
        # Measure epoch time
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            # For SAM optimizer
            if optimizer_name.lower() == 'sam':
                # Define closure for SAM
                def closure():
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    return loss
                
                # First SAM step: Compute gradients and perturb weights
                loss = optimizer.first_step(closure)
                
                # Second SAM step: Compute gradients at perturbed weights and update
                optimizer.second_step(closure)
                
                # Get outputs for accuracy calculation
                outputs = model(x)
            else:
                # Standard optimization
                optimizer.zero_grad()
                outputs = model(x)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()
            
            # Update metrics
            batch_size = x.data.shape[0]
            train_loss += float(loss.data) * batch_size
            
            # Calculate accuracy
            predictions = np.argmax(outputs.data, axis=1)
            y_indices = y.data.astype(np.int32)
            batch_correct = np.sum(predictions == y_indices)
            
            train_correct += batch_correct
            train_total += batch_size
        
        # Calculate training metrics
        avg_train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        for x, y in val_loader:
            outputs = model(x)
            loss = loss_fn(outputs, y)
            
            # Update metrics
            batch_size = x.data.shape[0]
            val_loss += float(loss.data) * batch_size
            
            # Calculate accuracy
            predictions = np.argmax(outputs.data, axis=1)
            y_indices = y.data.astype(np.int32)
            batch_correct = np.sum(predictions == y_indices)
            
            val_correct += batch_correct
            val_total += batch_size
        
        # Calculate validation metrics
        avg_val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total
        
        # Measure epoch time
        epoch_time = time.time() - epoch_start
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        history['epoch_times'].append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch}/{epochs} - "
              f"Time: {epoch_time:.2f}s - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Train Acc: {train_accuracy:.2%} - "
              f"Val Loss: {avg_val_loss:.4f} - "
              f"Val Acc: {val_accuracy:.2%}")
    
    return history


def evaluate_model(model, test_loader):
    """
    Evaluate model on test data.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        
    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    # Create loss function
    loss_fn = CrossEntropyLoss()
    
    # Set model to evaluation mode
    model.eval()
    
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    # Class-wise accuracy tracking
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    
    for x, y in test_loader:
        # Forward pass
        outputs = model(x)
        loss = loss_fn(outputs, y)
        
        # Update metrics
        batch_size = x.data.shape[0]
        test_loss += float(loss.data) * batch_size
        
        # Calculate accuracy
        predictions = np.argmax(outputs.data, axis=1)
        y_indices = y.data.astype(np.int32)
        
        # Update overall accuracy
        batch_correct = np.sum(predictions == y_indices)
        test_correct += batch_correct
        test_total += batch_size
        
        # Update class-wise accuracy
        for i, label in enumerate(y_indices):
            class_total[label] += 1
            if predictions[i] == label:
                class_correct[label] += 1
    
    # Calculate final metrics
    avg_test_loss = test_loss / test_total
    test_accuracy = test_correct / test_total
    
    # Calculate class-wise accuracy
    class_accuracy = np.zeros(10)
    for i in range(10):
        if class_total[i] > 0:
            class_accuracy[i] = class_correct[i] / class_total[i]
    
    return avg_test_loss, test_accuracy, class_accuracy


def run_optimizer_comparison():
    """
    Run a comparison of different optimizers on CIFAR-10.
    """
    # Load dataset
    X_train, X_val, X_test, y_train, y_val, y_test = load_cifar10_subset(n_samples=5000)
    
    # Create datasets and data loaders
    batch_size = 64
    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)
    test_dataset = Dataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Define optimizers to compare
    optimizers = ['SGD', 'Adam', 'SAM']
    
    # Training parameters
    epochs = 15
    learning_rate = 0.001
    
    # Results storage
    results = {}
    models = {}
    
    # Train with each optimizer
    for optimizer_name in optimizers:
        print(f"\n{'='*50}")
        print(f"Training with {optimizer_name} optimizer")
        print(f"{'='*50}")
        
        # Create a new model for each optimizer
        model = create_model()
        
        # Train the model
        history = train_and_evaluate(
            model, train_loader, val_loader, 
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            epochs=epochs
        )
        
        # Evaluate on test set
        test_loss, test_accuracy, class_accuracy = evaluate_model(model, test_loader)
        print(f"\nTest Results with {optimizer_name}:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.2%}")
        
        # Store results
        results[optimizer_name] = {
            'history': history,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'class_accuracy': class_accuracy
        }
        
        # Store model
        models[optimizer_name] = model
    
    # Plot comparison
    plot_results(results, optimizers)
    
    # Print final summary
    print("\nFinal comparison:")
    for optimizer_name in optimizers:
        print(f"{optimizer_name:<10}: Test Accuracy = {results[optimizer_name]['test_accuracy']:.2%}")
    
    # Return results for further analysis
    return results, models


def plot_results(results, optimizers):
    """
    Plot comparison of optimizer performance.
    
    Args:
        results: Dictionary with results for each optimizer
        optimizers: List of optimizer names
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for opt in optimizers:
        plt.plot(results[opt]['history']['train_loss'], label=opt)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation loss
    plt.subplot(2, 2, 2)
    for opt in optimizers:
        plt.plot(results[opt]['history']['val_loss'], label=opt)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training accuracy
    plt.subplot(2, 2, 3)
    for opt in optimizers:
        train_acc = [acc * 100 for acc in results[opt]['history']['train_acc']]
        plt.plot(train_acc, label=opt)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot validation accuracy
    plt.subplot(2, 2, 4)
    for opt in optimizers:
        val_acc = [acc * 100 for acc in results[opt]['history']['val_acc']]
        plt.plot(val_acc, label=opt)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison_results.png')
    print("\nResults plot saved to optimizer_comparison_results.png")
    
    # Plot class-wise accuracies
    plt.figure(figsize=(10, 6))
    width = 0.8 / len(optimizers)
    x = np.arange(10)  # 10 classes
    
    for i, opt in enumerate(optimizers):
        class_acc = results[opt]['class_accuracy'] * 100
        plt.bar(x + i * width - (len(optimizers)-1) * width / 2, class_acc, width, label=opt)
    
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Class-wise Accuracy by Optimizer')
    plt.xticks(x, [f'Class {i}' for i in range(10)])
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('class_accuracy_comparison.png')
    print("Class accuracy plot saved to class_accuracy_comparison.png")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the optimizer comparison
    results, models = run_optimizer_comparison()
