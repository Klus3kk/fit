import pytest
import numpy as np
from core.tensor import Tensor
from nn.linear import Linear
from nn.activations import ReLU, Softmax, Dropout
from nn.normalization import BatchNorm
from nn.sequential import Sequential


class TestLinear:
    def test_forward(self):
        # Create a simple linear layer
        linear = Linear(2, 3)

        # Fix the weights and biases for deterministic testing
        linear.weight.data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        linear.bias.data = np.array([0.1, 0.2, 0.3])

        # Test forward pass
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        out = linear(x)

        # Expected output:
        # [1.0, 2.0] @ [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] + [0.1, 0.2, 0.3]
        # = [0.9, 1.2, 1.5]
        # [3.0, 4.0] @ [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] + [0.1, 0.2, 0.3]
        # = [1.9, 2.4, 2.9]
        expected = np.array([[0.9, 1.2, 1.5], [1.9, 2.4, 2.9]])
        assert np.allclose(out.data, expected)

    def test_backward(self):
        linear = Linear(2, 3)
        linear.weight.data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        linear.bias.data = np.array([0.1, 0.2, 0.3])

        x = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        out = linear(x)

        # Set upstream gradient
        out.grad = np.array([[1.0, 1.0, 1.0]])
        out._backward()

        # Check gradients
        expected_x_grad = np.array([[0.6, 1.5]])  # x_grad = out.grad @ weight.T
        assert np.allclose(x.grad, expected_x_grad)

        expected_weight_grad = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ]).T  # weight_grad = x.T @ out.grad
        assert np.allclose(linear.weight.grad, expected_weight_grad)

        expected_bias_grad = np.array([1.0, 1.0, 1.0])  # bias_grad = out.grad.sum(axis=0)
        assert np.allclose(linear.bias.grad, expected_bias_grad)


class TestActivations:
    def test_relu_forward(self):
        relu = ReLU()
        x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]))
        out = relu(x)
        expected = np.array([0.0, 0.0, 1.0, 2.0])
        assert np.array_equal(out.data, expected)

    def test_relu_backward(self):
        relu = ReLU()
        x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]), requires_grad=True)
        out = relu(x)
        out.grad = np.array([1.0, 1.0, 1.0, 1.0])
        out._backward()

        expected_grad = np.array([0.0, 0.0, 1.0, 1.0])  # grad flows only through positive inputs
        assert np.array_equal(x.grad, expected_grad)

    def test_softmax_forward(self):
        softmax = Softmax()
        x = Tensor(np.array([[1.0, 2.0, 3.0]]))
        out = softmax(x)

        # Expected: e^[1,2,3] / sum(e^[1,2,3])
        expected = np.exp([1.0, 2.0, 3.0]) / np.sum(np.exp([1.0, 2.0, 3.0]))
        assert np.allclose(out.data[0], expected)
        assert np.isclose(np.sum(out.data), 1.0)  # Probabilities sum to 1

    def test_dropout(self):
        # Test in training mode
        dropout = Dropout(p=0.5)
        dropout.training = True

        x = Tensor(np.ones((10, 10)))
        out = dropout(x)

        # Check that some elements are zeroed out
        assert np.sum(out.data == 0) > 0
        # Check that non-zero elements are scaled
        assert np.all((out.data == 0) | (out.data > 1.0))

        # Test in evaluation mode
        dropout.training = False
        out = dropout(x)
        assert np.array_equal(out.data, x.data)  # No dropout in eval mode


class TestNormalization:
    def test_batch_norm_training(self):
        batch_norm = BatchNorm(3)
        batch_norm.training = True

        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        out = batch_norm(x)

        # In training mode, BatchNorm normalizes using batch statistics
        batch_mean = np.mean(x.data, axis=0)
        batch_var = np.var(x.data, axis=0)

        # Expected normalized data
        expected = (x.data - batch_mean) / np.sqrt(batch_var + batch_norm.eps)

        # With default gamma=1, beta=0
        assert np.allclose(out.data, expected)

        # Check that running stats are updated
        assert np.allclose(batch_norm.running_mean, batch_mean * batch_norm.momentum)
        assert np.allclose(batch_norm.running_var, batch_var * batch_norm.momentum + (1 - batch_norm.momentum))

    def test_batch_norm_inference(self):
        batch_norm = BatchNorm(3)
        batch_norm.training = False

        # Set running stats
        batch_norm.running_mean = np.array([1.0, 2.0, 3.0])
        batch_norm.running_var = np.array([1.0, 2.0, 3.0])

        x = Tensor(np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        out = batch_norm(x)

        # In inference mode, BatchNorm uses running statistics
        expected = (x.data - batch_norm.running_mean) / np.sqrt(batch_norm.running_var + batch_norm.eps)

        # With default gamma=1, beta=0
        assert np.allclose(out.data, expected)


class TestSequential:
    def test_forward(self):
        model = Sequential(
            Linear(2, 3),
            ReLU(),
            Linear(3, 1)
        )

        # Fix weights for deterministic testing
        model.layers[0].weight.data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        model.layers[0].bias.data = np.array([0.1, 0.2, 0.3])
        model.layers[2].weight.data = np.array([[0.7, 0.8, 0.9]])
        model.layers[2].bias.data = np.array([0.5])

        x = Tensor(np.array([[1.0, 2.0]]))
        out = model(x)

        # Manual calculation of expected output
        layer1_out = np.array([[0.9, 1.2, 1.5]])  # [1.0, 2.0] @ [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] + [0.1, 0.2, 0.3]
        relu_out = np.array([[0.9, 1.2, 1.5]])  # All positive, so no change
        expected = np.array([[relu_out @ np.array([[0.7], [0.8], [0.9]]) + 0.5]])  # layer2
        expected = expected.reshape(1, 1)

        assert np.allclose(out.data, expected)

    def test_parameters(self):
        model = Sequential(
            Linear(2, 3),
            ReLU(),
            Linear(3, 1)
        )

        params = model.parameters()

        # We expect 4 parameters: 2 weights and 2 biases
        assert len(params) == 4

        # Check if parameters are Tensor objects
        for param in params:
            assert isinstance(param, Tensor)