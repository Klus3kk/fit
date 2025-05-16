"""
Unit tests for the Sharpness-Aware Minimization (SAM) optimizer.
"""

import numpy as np
import pytest

from core.tensor import Tensor
from nn.linear import Linear
from nn.sequential import Sequential
from nn.activations import ReLU
from train.loss import MSELoss
from train.optim import SGD
from train.optim_sam import SAM


class TestSAMOptimizer:
    def setup_method(self):
        # Create a simple model
        self.model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))

        # Set fixed weights for deterministic testing
        self.model.layers[0].weight.data = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        )
        self.model.layers[0].bias.data = np.array([0.1, 0.1, 0.1, 0.1])
        self.model.layers[2].weight.data = np.array([[0.1, 0.2, 0.3, 0.4]])
        self.model.layers[2].bias.data = np.array([0.1])

        # Create data
        self.X = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        self.y = Tensor(np.array([[3.0], [7.0]]), requires_grad=False)

        # Create loss function
        self.loss_fn = MSELoss()

        # Base optimizer
        self.base_optimizer = SGD(self.model.parameters(), lr=0.1)

        # SAM optimizer
        self.sam_optimizer = SAM(self.model.parameters(), self.base_optimizer, rho=0.05)

    def test_sam_init(self):
        """Test SAM optimizer initialization."""
        assert self.sam_optimizer.rho == 0.05
        assert self.sam_optimizer.base_optimizer == self.base_optimizer
        assert len(self.sam_optimizer.parameters) == len(self.model.parameters())
        assert all(copy is None for copy in self.sam_optimizer.param_copies)

    def test_first_step(self):
        """Test the first step of SAM optimization."""

        # Define closure
        def closure():
            self.base_optimizer.zero_grad()
            output = self.model(self.X)
            loss = self.loss_fn(output, self.y)
            loss.backward()
            return loss

        # Store initial weights
        initial_weights = [param.data.copy() for param in self.model.parameters()]

        # Perform first step
        loss = self.sam_optimizer.first_step(closure)

        # Check that parameters have changed (perturbed)
        for i, param in enumerate(self.model.parameters()):
            assert not np.array_equal(param.data, initial_weights[i])

        # Check that param_copies contains the initial weights
        for i, copy in enumerate(self.sam_optimizer.param_copies):
            if copy is not None:
                assert np.array_equal(copy, initial_weights[i])

    def test_second_step(self):
        """Test the second step of SAM optimization."""

        # Define closure
        def closure():
            self.base_optimizer.zero_grad()
            output = self.model(self.X)
            loss = self.loss_fn(output, self.y)
            loss.backward()
            return loss

        # Perform first step
        self.sam_optimizer.first_step(closure)

        # Store perturbed weights
        perturbed_weights = [param.data.copy() for param in self.model.parameters()]

        # Perform second step
        self.sam_optimizer.second_step(closure)

        # Check that parameters have been restored and then updated
        for i, param in enumerate(self.model.parameters()):
            original = self.sam_optimizer.param_copies[i]
            if original is not None:
                # Parameters should no longer be the perturbed weights
                assert not np.array_equal(param.data, perturbed_weights[i])

                # Parameters should be updated from original (not equal to original)
                assert not np.array_equal(param.data, original)

    def test_complete_step(self):
        """Test a complete SAM optimization step."""

        # Define closure
        def closure():
            self.base_optimizer.zero_grad()
            output = self.model(self.X)
            loss = self.loss_fn(output, self.y)
            loss.backward()
            return loss

        # Store initial weights
        initial_weights = [param.data.copy() for param in self.model.parameters()]

        # Perform complete step
        loss = self.sam_optimizer.step(closure)

        # Check that parameters have changed
        for i, param in enumerate(self.model.parameters()):
            assert not np.array_equal(param.data, initial_weights[i])

    def test_grad_norm(self):
        """Test the gradient norm calculation."""
        # Set known gradients
        for i, param in enumerate(self.model.parameters()):
            param.grad = np.ones_like(param.data)

        # Calculate expected norm manually
        expected_norm = np.sqrt(
            sum(np.prod(param.data.shape) for param in self.model.parameters())
        )

        with_grad_params = [
            (i, param)
            for i, param in enumerate(self.model.parameters())
            if param.grad is not None
        ]

        # Get calculated norm
        norm = self.sam_optimizer._grad_norm(with_grad_params)

        # Check norm calculation
        assert np.isclose(norm, expected_norm)
