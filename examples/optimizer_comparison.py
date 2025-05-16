"""
Enhanced XOR problem comparison of different optimizers.

This version fixes convergence issues by:
1. Using Tanh activation instead of ReLU (better for XOR)
2. Using larger hidden layers (8 neurons)
3. Using better weight initialization for each optimizer
4. Adjusting learning rates and other hyperparameters
5. Adding visualization of decision boundaries
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


def train_xor_with_optimizer(optimizer_name, epochs=2000, verbose=True):
    """
    Train a model to solve the XOR problem with a specific optimizer.
    
    Args:
        optimizer_name: Name of the optimizer to use
        epochs: Number of training epochs
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training results
    """
    # Use a different seed for each optimizer to ensure different initialization
    np.random.seed(42 + hash(optimizer_name) % 1000)
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Convert to tensors
    X_tensor = Tensor(X, requires_grad=True)
    y_tensor = Tensor(y, requires_grad=False)
    
    # Create a model with adequate capacity for XOR
    # Using 8 hidden neurons with Tanh activation
    hidden_size = 8
    model = Sequential(Linear(2, hidden_size), Tanh(), Linear(hidden_size, 1))
    
    # Use better initializations based on the optimizer
    # For XOR, asymmetric initialization is crucial to break symmetry
    if optimizer_name == "SGD":
        model.layers[0].weight.data = np.random.uniform(-0.5, 0.5, (2, hidden_size))
        model.layers[0].bias.data = np.random.uniform(-0.1, 0.1, hidden_size)
        model.layers[2].weight.data = np.random.uniform(-0.5, 0.5, (hidden_size, 1))
        model.layers[2].bias.data = np.zeros(1)
    elif optimizer_name == "SGDMomentum":
        model.layers[0].weight.data = np.random.uniform(-0.7, 0.7, (2, hidden_size))
        model.layers[0].bias.data = np.random.uniform(-0.2, 0.2, hidden_size)
        model.layers[2].weight.data = np.random.uniform(-0.7, 0.7, (hidden_size, 1))
        model.layers[2].bias.data = np.zeros(1)
    elif optimizer_name == "Adam":
        model.layers[0].weight.data = np.random.uniform(-0.4, 0.4, (2, hidden_size))
        model.layers[0].bias.data = np.random.uniform(-0.1, 0.1, hidden_size)
        model.layers[2].weight.data = np.random.uniform(-0.4, 0.4, (hidden_size, 1))
        model.layers[2].bias.data = np.zeros(1)
    elif optimizer_name == "Lion":
        model.layers[0].weight.data = np.random.uniform(-1.0, 1.0, (2, hidden_size))
        model.layers[0].bias.data = np.random.uniform(-0.2, 0.2, hidden_size)
        model.layers[2].weight.data = np.random.uniform(-1.0, 1.0, (hidden_size, 1))
        model.layers[2].bias.data = np.zeros(1)
    elif optimizer_name == "SAM":
        model.layers[0].weight.data = np.random.uniform(-0.5, 0.5, (2, hidden_size))
        model.layers[0].bias.data = np.random.uniform(-0.1, 0.1, hidden_size)
        model.layers[2].weight.data = np.random.uniform(-0.5, 0.5, (hidden_size, 1))
        model.layers[2].bias.data = np.zeros(1)
        
    # Create loss function
    loss_fn = MSELoss()
    
    # Create optimizer with appropriate hyperparameters for each optimizer
    if optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=0.1)
    elif optimizer_name == "SGDMomentum":
        optimizer = SGDMomentum(model.parameters(), lr=0.05, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=0.03)
    elif optimizer_name == "Lion":
        optimizer = Lion(model.parameters(), lr=0.05)
    elif optimizer_name == "SAM":
        # SAM needs a base optimizer
        base_optimizer = Adam(model.parameters(), lr=0.02)
        optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
    # Training loop
    losses = []
    
    # Early stopping variables
    best_loss = float('inf')
    patience = 100
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Forward pass
        if optimizer_name == "SAM":
            # SAM requires a closure for both steps
            def closure():
                outputs = model(X_tensor)
                loss = loss_fn(outputs, y_tensor)
                loss.backward()
                return loss
                
            # First step (perturb weights)
            loss = optimizer.first_step(closure)
            
            # Second step (compute gradient at perturbed weights & update)
            optimizer.second_step(closure)
            
            losses.append(float(loss.data))
        else:
            # Standard optimization for other optimizers
            outputs = model(X_tensor)
            loss = loss_fn(outputs, y_tensor)
            losses.append(float(loss.data))
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
        
        # Early stopping check
        if loss.data < best_loss:
            best_loss = loss.data
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience and epoch > 300 and loss.data < 0.1:
            if verbose:
                print(f"{optimizer_name} - Early stopping at epoch {epoch}/{epochs}, Loss: {loss.data:.4f}")
            break
            
        # Print progress
        if verbose and epoch % 100 == 0:
            print(f"{optimizer_name} - Epoch {epoch}/{epochs}, Loss: {loss.data:.4f}")
            
    # Calculate accuracy and final predictions
    outputs = model(X_tensor).data
    predictions = outputs.flatten()
    predicted_classes = (predictions >= 0.5).astype(int)
    actual_classes = y.flatten()
    accuracy = np.mean(predicted_classes == actual_classes) * 100
    
    if verbose:
        print(f"\n{optimizer_name} final accuracy: {accuracy:.1f}%")
        print("Predictions vs. Actual:")
        for i in range(len(X)):
            input_data = X[i]
            actual = y[i][0]
            prediction = outputs[i][0]
            predicted_class = 1 if prediction >= 0.5 else 0
            print(f"Input: {input_data}, Predicted: {prediction:.4f} -> {predicted_class}, Actual: {actual}")
            
    # Return results in a dictionary
    return {
        "model": model,
        "losses": losses,
        "accuracy": accuracy,
        "predictions": predictions,
        "optimizer": optimizer_name,
    }


def plot_decision_boundaries(results, optimizers):
    """
    Plot decision boundaries for each optimizer.
    
    Args:
        results: Dictionary of optimizer results
        optimizers: List of optimizer names to plot
    """
    plt.figure(figsize=(15, 10))
    
    # Create a mesh grid for visualization
    h = 0.02
    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # XOR data points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]]).flatten()
    
    # Plot each optimizer's decision boundary
    for i, opt_name in enumerate(optimizers):
        if opt_name not in results:
            continue
            
        result = results[opt_name]
        model = result["model"]
        
        # Plot in a grid
        plt.subplot(2, 3, i + 1)
        
        # Make predictions on the mesh grid
        Z = []
        for point in mesh_points:
            x_point = Tensor(point.reshape(1, -1))
            pred = model(x_point).data[0, 0]
            Z.append(pred)
            
        # Reshape for contour plot
        Z = np.array(Z).reshape(xx.shape)
        
        # Plot decision boundary (0.5 threshold)
        plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=plt.cm.RdBu, alpha=0.5)
        plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-', linewidths=2)
        
        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors='k', s=80)
        
        # Mark correctly/incorrectly classified points
        preds = result["predictions"]
        pred_classes = (preds >= 0.5).astype(int)
        for j in range(4):
            if pred_classes[j] == y[j]:
                plt.plot(X[j, 0], X[j, 1], 'o', markersize=12, 
                         markerfacecolor='none', markeredgecolor='g', markeredgewidth=2)
            else:
                plt.plot(X[j, 0], X[j, 1], 'x', markersize=12, color='r', markeredgewidth=2)
                
        # Add title and accuracy
        plt.title(f"{opt_name} - Accuracy: {result['accuracy']:.1f}%")
        plt.xlabel("X1")
        plt.ylabel("X2")
        
    plt.tight_layout()
    plt.savefig("optimizer_decision_boundaries.png")
    print("\nDecision boundaries saved to optimizer_decision_boundaries.png")


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
    plt.yscale('log')  # Log scale to better see differences
    plt.ylim(1e-4, 2)  # Focus on relevant loss range
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
    print("\nComparison plot saved to optimizer_comparison.png")
    
    # Plot decision boundaries for each optimizer
    plot_decision_boundaries(results, optimizers)
    
    # Print summary
    print("\nOptimizer Performance Summary:")
    for opt in optimizers:
        print(f"- {opt}: {results[opt]['accuracy']:.1f}% accuracy, final loss = {results[opt]['losses'][-1]:.6f}")


if __name__ == "__main__":
    compare_optimizers()