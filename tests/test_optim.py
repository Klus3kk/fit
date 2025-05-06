import pytest
import numpy as np
from core.tensor import Tensor
from train.optim import SGD, SGDMomentum, Adam


class TestOptimizers:
    def test_sgd(self):
        # Create a parameter
        param = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        param.grad = np.array([0.1, 0.2, 0.3])

        # Create optimizer with learning rate 0.1
        optimizer = SGD([param], lr=0.1)

        # Take a step
        optimizer.step()

        # Check parameter update: param = param - lr * grad
        expected = np.array([0.99, 1.98, 2.97])
        assert np.allclose(param.data, expected)

        # Test zero_grad
        optimizer.zero_grad()
        assert param.grad is None

    def test_sgd_momentum(self):
        # Create a parameter
        param = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        param.grad = np.array([0.1, 0.2, 0.3])

        # Create optimizer with momentum 0.9
        optimizer = SGDMomentum([param], lr=0.1, momentum=0.9)

        # First step initializes momentum
        optimizer.step()

        # velocity = momentum * velocity - lr * grad = 0.9 * 0 - 0.1 * [0.1, 0.2, 0.3] = [-0.01, -0.02, -0.03]
        # param = param + velocity = [1.0, 2.0, 3.0] + [-0.01, -0.02, -0.03] = [0.99, 1.98, 2.97]
        expected = np.array([0.99, 1.98, 2.97])
        assert np.allclose(param.data, expected)
        assert np.allclose(optimizer.velocity[0], np.array([-0.01, -0.02, -0.03]))

        # Set new gradient and take another step
        param.grad = np.array([0.1, 0.2, 0.3])
        optimizer.step()

        # velocity = momentum * velocity - lr * grad = 0.9 * [-0.01, -0.02, -0.03] - 0.1 * [0.1, 0.2, 0.3]
        #          = [-0.009, -0.018, -0.027] - [0.01, 0.02, 0.03] = [-0.019, -0.038, -0.057]
        # param = param + velocity = [0.99, 1.98, 2.97] + [-0.019, -0.038, -0.057] = [0.971, 1.942, 2.913]
        expected_velocity = np.array([-0.019, -0.038, -0.057])
        assert np.allclose(optimizer.velocity[0], expected_velocity)
        expected_param = np.array([0.971, 1.942, 2.913])
        assert np.allclose(param.data, expected_param)

    def test_adam(self):
        # Create a parameter
        param = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        param.grad = np.array([0.1, 0.2, 0.3])

        # Create Adam optimizer
        optimizer = Adam([param], lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        # Take a step
        optimizer.step()

        # Check first moment (momentum)
        expected_m = np.array([0.01, 0.02, 0.03])  # m = (1-beta1) * grad
        assert np.allclose(optimizer.m[0], expected_m)

        # Check second moment
        expected_v = np.array([0.001, 0.004, 0.009])  # v = (1-beta2) * grad^2
        assert np.allclose(optimizer.v[0], expected_v)

        # Check bias correction
        # m_hat = m / (1 - beta1^t) = m / (1 - 0.9^1) = m / 0.1 = 10 * m
        # v_hat = v / (1 - beta2^t) = v / (1 - 0.999^1) = v / 0.001 = 1000 * v

        # Update: param = param - lr * m_hat / (sqrt(v_hat) + eps)
        # param = param - 0.001 * (10 * m) / (sqrt(1000 * v) + 1e-8)

        # Let's manually calculate the expected param values
        m_hat = expected_m / 0.1
        v_hat = expected_v / 0.001
        update = 0.001 * m_hat / (np.sqrt(v_hat) + 1e-8)
        expected_param = np.array([1.0, 2.0, 3.0]) - update

        assert np.allclose(param.data, expected_param)