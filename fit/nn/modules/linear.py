"""
Implementation of linear (fully connected) layers.
"""

import numpy as np

from fit.core.tensor import Tensor
from fit.nn.modules.base import Layer


class Linear(Layer):
    """
    Linear (fully connected) layer.

    Applies a linear transformation: y = xW^T + b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias term
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize weights with He initialization
        std = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.normal(0, std, (out_features, in_features)), requires_grad=True
        )

        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        else:
            self.bias = None

        # Register parameters
        self.add_parameter(self.weight)
        if self.bias is not None:
            self.add_parameter(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of linear layer.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Compute x @ W^T
        output = x @ self.weight.T

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"

    def get_config(self):
        """Get layer configuration for serialization."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.use_bias,
        }


class Bilinear(Layer):
    """
    Bilinear layer: y = x1^T @ W @ x2 + b
    """

    def __init__(
        self, in1_features: int, in2_features: int, out_features: int, bias: bool = True
    ):
        """
        Initialize bilinear layer.

        Args:
            in1_features: Size of first input
            in2_features: Size of second input
            out_features: Size of output
            bias: Whether to include bias
        """
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize weight tensor (out_features, in1_features, in2_features)
        std = np.sqrt(1.0 / (in1_features * in2_features))
        self.weight = Tensor(
            np.random.normal(0, std, (out_features, in1_features, in2_features)),
            requires_grad=True,
        )

        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        else:
            self.bias = None

        self.add_parameter(self.weight)
        if self.bias is not None:
            self.add_parameter(self.bias)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Forward pass of bilinear layer.

        Args:
            x1: First input tensor (batch_size, in1_features)
            x2: Second input tensor (batch_size, in2_features)

        Returns:
            Output tensor (batch_size, out_features)
        """
        batch_size = x1.data.shape[0]

        # Compute bilinear transformation
        # For each output feature: x1^T @ W[i] @ x2
        output_data = np.zeros((batch_size, self.out_features))

        for i in range(self.out_features):
            # x1 @ W[i] @ x2 for each sample in batch
            for b in range(batch_size):
                output_data[b, i] = x1.data[b] @ self.weight.data[i] @ x2.data[b]

        output = Tensor(
            output_data,
            requires_grad=(
                x1.requires_grad or x2.requires_grad or self.weight.requires_grad
            ),
        )

        def _backward():
            if output.grad is None:
                return

            # Gradients w.r.t. inputs and weights
            if x1.requires_grad:
                x1_grad = np.zeros_like(x1.data)
                for b in range(batch_size):
                    for i in range(self.out_features):
                        x1_grad[b] += output.grad[b, i] * (
                            self.weight.data[i] @ x2.data[b]
                        )
                x1.grad = x1_grad if x1.grad is None else x1.grad + x1_grad

            if x2.requires_grad:
                x2_grad = np.zeros_like(x2.data)
                for b in range(batch_size):
                    for i in range(self.out_features):
                        x2_grad[b] += output.grad[b, i] * (
                            self.weight.data[i].T @ x1.data[b]
                        )
                x2.grad = x2_grad if x2.grad is None else x2.grad + x2_grad

            if self.weight.requires_grad:
                weight_grad = np.zeros_like(self.weight.data)
                for b in range(batch_size):
                    for i in range(self.out_features):
                        weight_grad[i] += output.grad[b, i] * np.outer(
                            x1.data[b], x2.data[b]
                        )
                self.weight.grad = (
                    weight_grad
                    if self.weight.grad is None
                    else self.weight.grad + weight_grad
                )

        output._backward = _backward
        output._prev = {x1, x2, self.weight}

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output


class Identity(Layer):
    """
    Identity layer - returns input unchanged.
    Useful for skip connections and as placeholder.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x


class Flatten(Layer):
    """
    Flatten layer - reshapes input to 2D tensor.
    """

    def __init__(self, start_dim: int = 1):
        """
        Initialize flatten layer.

        Args:
            start_dim: Dimension to start flattening from
        """
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Flatten input tensor.

        Args:
            x: Input tensor

        Returns:
            Flattened tensor
        """
        batch_size = x.data.shape[0]
        # Flatten all dimensions after start_dim
        new_shape = (batch_size, -1)
        return x.reshape(new_shape)


class Embedding(Layer):
    """
    Embedding layer for discrete tokens.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int = None
    ):
        """
        Initialize embedding layer.

        Args:
            num_embeddings: Size of dictionary of embeddings
            embedding_dim: Size of each embedding vector
            padding_idx: If given, pads output with embedding vector at padding_idx
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Initialize embedding matrix
        self.weight = Tensor(
            np.random.normal(0, 1, (num_embeddings, embedding_dim)), requires_grad=True
        )

        # Zero out padding embedding if specified
        if padding_idx is not None:
            self.weight.data[padding_idx] = 0

        self.add_parameter(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Look up embeddings.

        Args:
            x: Tensor containing indices

        Returns:
            Embedded tensor
        """
        # Convert indices to integers
        indices = x.data.astype(int)

        # Look up embeddings
        embedded_data = self.weight.data[indices]

        output = Tensor(embedded_data, requires_grad=self.weight.requires_grad)

        def _backward():
            if output.grad is None or not self.weight.requires_grad:
                return

            # Accumulate gradients for each embedding
            weight_grad = np.zeros_like(self.weight.data)
            flat_indices = indices.flatten()
            flat_grad = output.grad.reshape(-1, self.embedding_dim)

            for i, idx in enumerate(flat_indices):
                weight_grad[idx] += flat_grad[i]

            # Zero out gradient for padding index
            if self.padding_idx is not None:
                weight_grad[self.padding_idx] = 0

            self.weight.grad = (
                weight_grad
                if self.weight.grad is None
                else self.weight.grad + weight_grad
            )

        output._backward = _backward
        output._prev = {self.weight}

        return output
