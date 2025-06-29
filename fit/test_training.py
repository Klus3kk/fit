"""
Test training functionality: optimizers, loss functions, and training loop.
"""

import numpy as np
from fit.core.tensor import Tensor
from fit.nn.modules.linear import Linear
from fit.nn.modules.activation import ReLU, Softmax
from fit.nn.modules.container import Sequential
from fit.optim.adam import Adam
from fit.optim.sgd import SGD
from fit.loss.classification import CrossEntropyLoss
from fit.loss.regression import MSELoss


def test_optimizers():
    """Test optimizer functionality."""
    print("=== Testing Optimizers ===")

    # Create a simple model
    model = Sequential(Linear(2, 3), ReLU(), Linear(3, 1))

    # Test Adam optimizer
    print("Testing Adam optimizer...")
    adam = Adam(model.parameters(), lr=0.01)

    # Test SGD optimizer
    print("Testing SGD optimizer...")
    sgd = SGD(model.parameters(), lr=0.01)

    print("Optimizers created successfully!")
    return True


def test_loss_functions():
    """Test loss functions."""
    print("\n=== Testing Loss Functions ===")

    # Test MSE Loss
    print("Testing MSE Loss...")
    mse = MSELoss()

    # Create some dummy data
    predictions = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    targets = Tensor([[1.5, 2.5], [2.5, 3.5]])

    loss = mse(predictions, targets)
    print(f"MSE Loss: {loss.data}")

    # Test CrossEntropy Loss
    print("Testing CrossEntropy Loss...")
    ce = CrossEntropyLoss()

    # Create logits and targets
    logits = Tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.8]], requires_grad=True)
    targets = Tensor([0, 1])  # Class indices

    loss = ce(logits, targets)
    print(f"CrossEntropy Loss: {loss.data}")

    return True


def test_simple_training():
    """Test a simple training loop."""
    print("\n=== Testing Simple Training Loop ===")

    # Create XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)  # XOR truth table

    print("XOR Dataset:")
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Output: {y[i]}")

    # Create model
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 2), Softmax())

    # Create optimizer and loss
    optimizer = Adam(model.parameters(), lr=0.1)
    loss_fn = CrossEntropyLoss()

    print("\nStarting training...")

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(len(X)):
            # Forward pass
            x = Tensor([X[i]], requires_grad=True)
            target = Tensor([y[i]])

            output = model(x)
            loss = loss_fn(output, target)

            # Zero gradients
            for param in model.parameters():
                param.grad = None

            # Backward pass
            try:
                loss.backward()

                # Update parameters
                optimizer.step()

                total_loss += loss.data
            except Exception as e:
                print(f"Error in backward pass: {e}")
                return False

        if epoch % 2 == 0:
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

    # Test the trained model
    print("\nTesting trained model:")
    for i in range(len(X)):
        x = Tensor([X[i]], requires_grad=False)
        output = model(x)
        predicted = np.argmax(output.data)
        print(f"Input: {X[i]} -> Predicted: {predicted}, Actual: {y[i]}")

    return True


def test_data_utilities():
    """Test data utilities."""
    print("\n=== Testing Data Utilities ===")

    try:
        from fit.data.dataset import Dataset
        from fit.data.dataloader import DataLoader

        # Create simple dataset
        X = np.random.randn(20, 3)
        y = np.random.randint(0, 2, 20)

        dataset = Dataset(X, y)
        print(f"Dataset created with {len(dataset)} samples")

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        print(f"DataLoader created with batch_size=4")

        # Test iteration
        print("Testing DataLoader iteration:")
        for i, (batch_x, batch_y) in enumerate(dataloader):
            print(
                f"Batch {i}: X shape {batch_x.data.shape}, y shape {batch_y.data.shape}"
            )
            if i >= 2:  # Only show first 3 batches
                break

        return True
    except ImportError as e:
        print(f"Data utilities not available: {e}")
        return False


if __name__ == "__main__":
    try:
        test_optimizers()
        test_loss_functions()
        test_simple_training()
        test_data_utilities()
        print("\n✅ Training tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
