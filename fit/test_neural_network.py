"""
Test neural network components: layers, activations, and models.
"""

import numpy as np
from fit.core.tensor import Tensor
from fit.nn.modules.linear import Linear
from fit.nn.modules.activation import ReLU, Softmax
from fit.nn.modules.container import Sequential


def test_linear_layer():
    """Test linear layer functionality."""
    print("=== Testing Linear Layer ===")

    # Create a simple linear layer
    layer = Linear(3, 2)  # 3 inputs, 2 outputs

    print(f"Weight shape: {layer.weight.data.shape}")
    print(f"Bias shape: {layer.bias.data.shape}")
    print(f"Weights:\n{layer.weight.data}")
    print(f"Bias: {layer.bias.data}")

    # Test forward pass
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    print(f"Input shape: {x.data.shape}")
    print(f"Input:\n{x.data}")

    output = layer(x)
    print(f"Output shape: {output.data.shape}")
    print(f"Output:\n{output.data}")

    return True


def test_activations():
    """Test activation functions."""
    print("\n=== Testing Activation Functions ===")

    # Test ReLU
    relu = ReLU()
    x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], requires_grad=True)

    print(f"Input: {x.data}")
    relu_out = relu(x)
    print(f"ReLU output: {relu_out.data}")

    # Test Softmax
    softmax = Softmax()
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    print(f"Softmax input: {x.data}")
    softmax_out = softmax(x)
    print(f"Softmax output: {softmax_out.data}")
    print(f"Sum: {softmax_out.data.sum()} (should be ~1.0)")

    return True


def test_sequential_model():
    """Test sequential model."""
    print("\n=== Testing Sequential Model ===")

    # Create a simple 2-layer network
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 3), Softmax())

    print("Model created successfully!")
    print(f"Number of layers: {len(model.layers)}")

    # Test forward pass
    x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    print(f"Input: {x.data}")

    output = model(x)
    print(f"Model output: {output.data}")
    print(f"Output sum: {output.data.sum()} (should be ~1.0)")

    # Test with batch
    x_batch = Tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]], requires_grad=True)
    print(f"Batch input shape: {x_batch.data.shape}")

    batch_output = model(x_batch)
    print(f"Batch output shape: {batch_output.data.shape}")
    print(f"Batch output:\n{batch_output.data}")

    return True


def test_parameters():
    """Test parameter extraction."""
    print("\n=== Testing Parameter Extraction ===")

    model = Sequential(Linear(2, 3), ReLU(), Linear(3, 1))

    params = model.parameters()
    print(f"Number of parameters: {len(params)}")

    for i, param in enumerate(params):
        print(
            f"Parameter {i}: shape {param.data.shape}, requires_grad: {param.requires_grad}"
        )

    return True


if __name__ == "__main__":
    try:
        test_linear_layer()
        test_activations()
        test_sequential_model()
        test_parameters()
        print("\n✅ Neural network tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
