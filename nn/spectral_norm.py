"""
Implementation of Spectral Normalization for neural network layers.

Spectral Normalization controls the Lipschitz constant of the network by
constraining the spectral norm (maximum singular value) of each weight matrix.
This helps with training stability, especially for challenging problems
like the XOR task, and is also important for GANs and robust networks.

Paper: https://arxiv.org/abs/1802.05957
"""

import numpy as np
from typing import List, Optional, Tuple, Union

from core.tensor import Tensor
from nn.layer import Layer


class SpectralNorm:
    """
    Spectral Normalization for weight matrices.

    This class implements the power iteration method to estimate
    the spectral norm (largest singular value) of a weight matrix
    and then normalizes the weights by this value.
    """

    def __init__(self, weight: Tensor, n_power_iterations: int = 1, eps: float = 1e-12):
        """
        Initialize Spectral Normalization.

        Args:
            weight: Weight tensor to normalize
            n_power_iterations: Number of power iterations for estimating spectral norm
            eps: Small constant for numerical stability
        """
        self.weight = weight
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        # Get weight shape
        weight_mat = self._reshape_weight_to_matrix()
        height, width = weight_mat.shape

        # Initialize u and v vectors for power iteration
        # Using random normal initialization and normalization
        u_init = np.random.normal(0, 1, (1, height))
        u_init = u_init / np.sqrt(np.sum(u_init**2)) + self.eps
        self.u = u_init

        v_init = np.random.normal(0, 1, (width, 1))
        v_init = v_init / np.sqrt(np.sum(v_init**2)) + self.eps
        self.v = v_init

        # Initialize sigma (spectral norm)
        self.sigma = None

    def _reshape_weight_to_matrix(self) -> np.ndarray:
        """
        Reshape the weight tensor to a matrix.

        For Linear layers: (in_features, out_features)
        For Conv2d layers: (out_channels, in_channels * kernel_height * kernel_width)

        Returns:
            Reshaped weight matrix
        """
        weight = self.weight.data

        # For now, we only support 2D weight matrices
        if weight.ndim == 2:
            return weight
        else:
            # Reshape to 2D matrix
            return weight.reshape(weight.shape[0], -1)

    def _power_iteration(self, weight_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform power iteration to estimate the spectral norm.

        Args:
            weight_mat: Reshaped weight matrix

        Returns:
            Tuple of (u, v) vectors
        """
        u = self.u
        v = self.v

        for _ in range(self.n_power_iterations):
            # v = W^T @ u / ||W^T @ u||
            v_new = u @ weight_mat
            v_new = v_new / (np.sqrt(np.sum(v_new**2)) + self.eps)

            # u = W @ v / ||W @ v||
            u_new = weight_mat @ v_new.T
            u_new = u_new.T / (np.sqrt(np.sum(u_new**2)) + self.eps)

            u = u_new
            v = v_new

        # Update stored u and v
        self.u = u
        self.v = v

        return u, v

    def compute_weight(self) -> Tensor:
        """
        Compute the spectrally normalized weight.

        Returns:
            Normalized weight tensor
        """
        # Reshape weight to matrix
        weight_mat = self._reshape_weight_to_matrix()

        # Perform power iteration
        u, v = self._power_iteration(weight_mat)

        # Compute sigma (spectral norm)
        # sigma = u @ W @ v
        sigma = (u @ weight_mat @ v.T)[0, 0]
        self.sigma = sigma

        # Normalize weight
        normalized_weight = self.weight.data / sigma

        # Create normalized weight tensor
        normalized_weight_tensor = Tensor(
            normalized_weight, requires_grad=self.weight.requires_grad
        )

        # Forward derivative information
        def _backward():
            if self.weight.requires_grad and normalized_weight_tensor.grad is not None:
                # Compute gradient with respect to original weight
                # dL/dW = dL/dW_normalized * d(W_normalized)/dW
                # d(W_normalized)/dW = 1/sigma * (I - (u @ v^T @ W) / (u @ W @ v))

                # Simplified: we approximate the gradient as 1/sigma * dL/dW_normalized
                grad = normalized_weight_tensor.grad / sigma

                # Update the original weight's gradient
                self.weight.grad = (
                    grad if self.weight.grad is None else self.weight.grad + grad
                )

        # Set up backward function and dependencies
        normalized_weight_tensor._backward = _backward
        normalized_weight_tensor._prev = {self.weight}

        return normalized_weight_tensor


class SpectralNormLayer(Layer):
    """
    Layer wrapper that applies Spectral Normalization to weights.

    This can be used to wrap any layer with weights (e.g., Linear, Conv2d)
    to enforce spectral normalization during forward passes.
    """

    def __init__(self, layer: Layer, n_power_iterations: int = 1, eps: float = 1e-12):
        """
        Initialize Spectral Normalization layer.

        Args:
            layer: Layer to apply spectral normalization to
            n_power_iterations: Number of power iterations for estimating spectral norm
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.layer = layer

        # Find all weight parameters
        self.weight_params = []
        self.sn_modules = {}

        # If it's a Linear layer, we know it has 'weight' attribute
        if hasattr(layer, "weight"):
            # Initialize spectral norm module for the weight
            self.sn_modules["weight"] = SpectralNorm(
                layer.weight, n_power_iterations, eps
            )
            self.weight_params.append("weight")

        # Add child layers
        self.add_child(layer)

    def forward(self, *args, **kwargs):
        """
        Forward pass with spectral normalization applied.

        Args:
            *args: Positional arguments for the wrapped layer
            **kwargs: Keyword arguments for the wrapped layer

        Returns:
            Output of the wrapped layer with normalized weights
        """
        # Replace original weights with spectrally normalized weights
        original_weights = {}
        for name in self.weight_params:
            original_weights[name] = getattr(self.layer, name)
            setattr(self.layer, name, self.sn_modules[name].compute_weight())

        # Forward pass through the wrapped layer
        out = self.layer(*args, **kwargs)

        # Restore original weights
        for name, weight in original_weights.items():
            setattr(self.layer, name, weight)

        return out

    def parameters(self):
        """Get parameters of the wrapped layer."""
        return self.layer.parameters()


class SpectralNormLinear(SpectralNormLayer):
    """
    Convenience class for a Linear layer with Spectral Normalization.
    """

    def __init__(
        self, in_features, out_features, bias=True, n_power_iterations=1, eps=1e-12
    ):
        """
        Initialize a spectrally normalized Linear layer.

        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: Whether to include bias
            n_power_iterations: Number of power iterations
            eps: Numerical stability constant
        """
        from nn.linear import Linear

        # Create regular Linear layer
        linear = Linear(in_features, out_features)

        # Apply spectral normalization
        super().__init__(linear, n_power_iterations, eps)
