"""
Basic debugging/diagnostic tool for neural networks.

This script helps diagnose and fix common issues with neural network training,
particularly focusing on the XOR problem which can be tricky to learn.
"""

import numpy as np

from core.tensor import Tensor
from nn.activations import ReLU, Tanh
from nn.linear import Linear
from nn.sequential import Sequential
from train.loss import MSELoss
from train.optim import SGD, Adam


def diagnose_xor_training():
    """
    Run diagnostic tests for XOR training to identify common issues.
    """
    print("===== XOR Problem Neural Network Diagnostics =====")

    # Set up the XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Tensor conversion
    X_tensor = Tensor(X, requires_grad=True)
    y_tensor = Tensor(y, requires_grad=False)

    # Test 1: Fully connected network with ReLU
    print("\nTest 1: Two-layer network with ReLU activation")
    model1 = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    )
    # Random initialization
    np.random.seed(42)
    model1.layers[0].weight.data = np.random.randn(2, 4) * 0.1
    model1.layers[0].bias.data = np.zeros(4)

    # Forward pass
    outputs1 = model1(X_tensor)
    print("Initial predictions:")
    print(outputs1.data)

    # Train briefly
    loss_fn = MSELoss()
    optimizer1 = SGD(model1.parameters(), lr=0.1)

    losses1 = []
    for epoch in range(100):
        outputs = model1(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        losses1.append(loss.data)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer1.step()
        optimizer1.zero_grad()

    # Check final predictions
    final_outputs1 = model1(X_tensor)
    print("\nFinal predictions after 100 epochs (ReLU):")
    print(final_outputs1.data)
    print(f"Final loss: {losses1[-1]:.4f}")

    # Test 2: Network with Tanh activation
    print("\nTest 2: Two-layer network with Tanh activation")
    model2 = Sequential(
        Linear(2, 4),
        Tanh(),
        Linear(4, 1)
    )
    # Better initialization for Tanh
    np.random.seed(42)
    model2.layers[0].weight.data = np.array([
        [1.0, -1.0, 0.5, -0.5],
        [-1.0, 1.0, -0.5, 0.5]
    ])
    model2.layers[0].bias.data = np.array([0.1, 0.1, -0.1, -0.1])

    # Forward pass
    outputs2 = model2(X_tensor)
    print("Initial predictions:")
    print(outputs2.data)

    # Train briefly
    optimizer2 = Adam(model2.parameters(), lr=0.1)

    losses2 = []
    for epoch in range(100):
        outputs = model2(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        losses2.append(loss.data)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer2.step()
        optimizer2.zero_grad()

    # Check final predictions
    final_outputs2 = model2(X_tensor)
    print("\nFinal predictions after 100 epochs (Tanh):")
    print(final_outputs2.data)
    print(f"Final loss: {losses2[-1]:.4f}")

    # Test 3: Gradient check
    print("\nTest 3: Gradient check for Tanh network")

    # Fresh model
    model3 = Sequential(
        Linear(2, 2),
        Tanh(),
        Linear(2, 1)
    )

    # One forward and backward pass
    outputs3 = model3(X_tensor)
    loss3 = loss_fn(outputs3, y_tensor)
    loss3.backward()

    # Check if gradients are being computed
    all_params = model3.parameters()
    grad_norms = []
    for i, param in enumerate(all_params):
        if param.grad is not None:
            grad_norm = np.sqrt(np.sum(param.grad ** 2))
            grad_norms.append(grad_norm)
            print(f"Parameter {i} gradient norm: {grad_norm:.6f}")
        else:
            print(f"Parameter {i} has no gradient!")

    if len(grad_norms) > 0:
        avg_norm = np.mean(grad_norms)
        print(f"Average gradient norm: {avg_norm:.6f}")
        if avg_norm < 1e-6:
            print("WARNING: Gradients are very small, may indicate vanishing gradient!")
    else:
        print("ERROR: No gradients are being computed!")

    # Test 4: Layer-by-layer activation visualization
    print("\nTest 4: Layer-by-layer activations")

    # Create a model with Tanh
    model4 = Sequential(
        Linear(2, 3),
        Tanh(),
        Linear(3, 1)
    )

    # Proper initialization
    model4.layers[0].weight.data = np.array([
        [1.0, -1.0, 0.5],
        [-1.0, 1.0, -0.5]
    ])
    model4.layers[0].bias.data = np.array([0.1, 0.1, -0.1])

    # Get first layer output (pre-activation)
    x_input = X_tensor
    first_layer_output = np.zeros((4, 3))
    for i in range(4):
        xi = Tensor(X[i:i + 1])
        # Linear layer output before activation
        out_i = model4.layers[0](xi)
        first_layer_output[i] = out_i.data

    print("First layer output (pre-activation):")
    print(first_layer_output)

    # Get first layer output (post-activation)
    post_activation = np.zeros((4, 3))
    for i in range(4):
        xi = Tensor(X[i:i + 1])
        out_i = model4.layers[0](xi)
        post_i = model4.layers[1](out_i)
        post_activation[i] = post_i.data

    print("\nFirst layer output (post-activation):")
    print(post_activation)

    # Check if the neurons are capable of representing XOR
    print("\nAnalyzing neuron activations for XOR capability:")

    # For neurons to solve XOR, they need to be activated differently for each input pattern
    activation_patterns = np.round(post_activation, 2)
    unique_patterns = np.unique(activation_patterns, axis=0)

    print(f"Number of unique activation patterns: {len(unique_patterns)}")

    if len(unique_patterns) >= 3:
        print("✓ Neurons can represent XOR (at least 3 unique activation patterns)")
    else:
        print("✗ Neurons may struggle to represent XOR (need more unique activation patterns)")

    # Summary and recommendations
    print("\n===== Diagnostic Summary =====")
    if losses1[-1] < 0.25:
        print("✓ ReLU network successfully learning XOR")
    else:
        print("✗ ReLU network struggling to learn XOR")

    if losses2[-1] < 0.25:
        print("✓ Tanh network successfully learning XOR")
    else:
        print("✗ Tanh network struggling to learn XOR")

    print("\nRecommendations:")
    if losses1[-1] > 0.25 and losses2[-1] > 0.25:
        print("1. Use larger hidden layers (8-16 neurons)")
        print("2. Try different initializations to break symmetry")
        print("3. Use Adam optimizer with learning rate ~0.03-0.1")
        print("4. Train for more epochs (1000+)")
        print("5. Consider using Tanh activation with proper weight initialization")
    else:
        print("Your network appears to be capable of learning XOR. For best results:")
        print("1. Use Tanh activation for the XOR problem")
        print("2. Initialize weights to break symmetry")
        print("3. Use Adam optimizer for faster convergence")


if __name__ == "__main__":
    diagnose_xor_training()
