"""
This module implements the Tensor class for automatic differentiation in the ML framework.
"""

import numpy as np
from typing import Any, Callable, List, Optional, Set, Tuple, Union

from core.autograd import Node, get_function


class Tensor(Node):
    """
    Core tensor class with automatic differentiation capabilities.

    Implements a computational graph with automatic gradient calculation
    for building and training neural networks.
    """

    def __init__(self, data, requires_grad=False):
        """
        Initialize a new Tensor.

        Args:
            data: Input data (numpy array or convertible to numpy array)
            requires_grad: Whether the tensor requires gradient computation
        """
        super().__init__(requires_grad=requires_grad)

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)  # Use float64 for better precision
        else:
            # Ensure data is float type for gradient calculations
            if data.dtype.kind != "f":
                data = data.astype(np.float64)

        self.data = data
        self._prev = set()  # Dependencies for backward pass
        self._backward = lambda: None  # Default backward function

    def __repr__(self):
        """Return string representation of the tensor."""
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad})"

    def __add__(self, other):
        """
        Add two tensors or a tensor and a scalar.

        Args:
            other: Another tensor or scalar value

        Returns:
            A new tensor containing the result
        """
        # Convert scalar to tensor if needed
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Use the Add function from autograd
        add_fn = get_function("add")
        return add_fn.forward(self, other)

    def __radd__(self, other):
        """Handle right addition (scalar + tensor)."""
        return self.__add__(other)

    def __mul__(self, other):
        """
        Multiply two tensors or a tensor and a scalar.

        Args:
            other: Another tensor or scalar value

        Returns:
            A new tensor containing the result
        """
        # Convert scalar to tensor if needed
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Use the Multiply function from autograd
        mul_fn = get_function("multiply")
        return mul_fn.forward(self, other)

    def __rmul__(self, other):
        """Handle right multiplication (scalar * tensor)."""
        return self.__mul__(other)

    def __sub__(self, other):
        """
        Subtract a tensor or scalar from this tensor.

        Args:
            other: Another tensor or scalar value

        Returns:
            A new tensor containing the result
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        """Handle right subtraction (scalar - tensor)."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + (-self)

    def __neg__(self):
        """
        Negate this tensor.

        Returns:
            A new tensor with negated values
        """
        return self * -1

    def __truediv__(self, other):
        """
        Divide this tensor by another tensor or scalar.

        Args:
            other: Another tensor or scalar value

        Returns:
            A new tensor containing the result
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other**-1

    def __rtruediv__(self, other):
        """Handle right division (scalar / tensor)."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * self**-1

    def __pow__(self, power):
        """
        Raise this tensor to a power.

        Args:
            power: Exponent value

        Returns:
            A new tensor containing the result
        """
        if isinstance(power, Tensor):
            raise NotImplementedError("Tensor powers not yet supported")

        if not self.requires_grad:
            return Tensor(self.data**power)

        # For now, handle this directly since we haven't defined a Power function
        out = Tensor(self.data**power, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad and out.grad is not None:
                grad = power * self.data ** (power - 1) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = {self}
        return out

    def __matmul__(self, other):
        """
        Perform matrix multiplication with another tensor.

        Args:
            other: Another tensor for matrix multiplication

        Returns:
            A new tensor containing the result
        """
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Use the MatMul function from autograd
        matmul_fn = get_function("matmul")
        return matmul_fn.forward(self, other)

    def __getitem__(self, idx):
        """
        Get items from the tensor at specified indices.

        Args:
            idx: Index or slice to retrieve

        Returns:
            A new tensor with the selected items
        """
        out = Tensor(self.data[idx], requires_grad=self.requires_grad)

        if self.requires_grad:

            def _backward():
                if out.grad is not None:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    self.grad[idx] += out.grad

            out._backward = _backward
            out._prev = {self}

        return out

    def __hash__(self):
        """Hash function for tensor objects."""
        return id(self)

    def __eq__(self, other):
        """Check if two tensor objects are the same."""
        return id(self) == id(other)

    def sum(self, axis=None, keepdims=False):
        """
        Sum tensor elements along specified axis.

        Args:
            axis: Axis along which to sum
            keepdims: Whether to keep the summed dimensions

        Returns:
            A new tensor with summed values
        """
        # Use the Sum function from autograd
        sum_fn = get_function("sum")
        return sum_fn.forward(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        """
        Calculate the mean of tensor elements along specified axis.

        Args:
            axis: Axis along which to calculate mean
            keepdims: Whether to keep the reduced dimensions

        Returns:
            A new tensor containing the mean value
        """
        # Use the Mean function from autograd
        mean_fn = get_function("mean")
        return mean_fn.forward(self, axis, keepdims)

    def exp(self):
        """
        Calculate the exponential of all tensor elements.

        Returns:
            A new tensor with exponential values
        """
        # Use the Exp function from autograd
        exp_fn = get_function("exp")
        return exp_fn.forward(self)

    def log(self):
        """
        Calculate the natural logarithm of all tensor elements.

        Returns:
            A new tensor with logarithm values
        """
        # Use the Log function from autograd
        log_fn = get_function("log")
        return log_fn.forward(self)

    def reshape(self, shape):
        """
        Reshape the tensor to the specified shape.

        Args:
            shape: New shape for the tensor

        Returns:
            A new tensor with the specified shape
        """
        # Use the Reshape function from autograd
        reshape_fn = get_function("reshape")
        return reshape_fn.forward(self, shape)

    def max(self, axis=None, keepdims=False):
        """
        Find maximum values along specified axis.

        Args:
            axis: Axis along which to find maximum values
            keepdims: Whether to keep the dimensions

        Returns:
            A new tensor with maximum values
        """
        data = self.data
        max_data = np.max(data, axis=axis, keepdims=keepdims)
        out = Tensor(max_data, requires_grad=self.requires_grad)

        if self.requires_grad:

            def _backward():
                if out.grad is not None:
                    # Create a mask for the maximum values
                    mask = np.zeros_like(self.data, dtype=np.float64)

                    if axis is None:
                        # Single global maximum
                        mask.flat[np.argmax(self.data)] = 1.0
                    else:
                        # Maximum along specified axis
                        indices = np.argmax(self.data, axis=axis)
                        if not keepdims:
                            # Create indices for all dimensions
                            idx_tuple = []
                            for i in range(self.data.ndim):
                                if i == axis:
                                    idx_tuple.append(indices)
                                else:
                                    idx_tuple.append(
                                        np.arange(self.data.shape[i])[:, None]
                                    )
                            # Set corresponding elements to 1.0
                            mask[tuple(idx_tuple)] = 1.0
                        else:
                            # Handle keepdims case
                            mask = self.data == np.max(
                                self.data, axis=axis, keepdims=True
                            )

                    # Apply upstream gradient
                    grad = mask * (
                        np.reshape(out.grad, mask.shape) if keepdims else out.grad
                    )
                    self.grad = grad if self.grad is None else self.grad + grad

            out._backward = _backward
            out._prev = {self}

        return out

    def relu(self):
        """
        Apply ReLU (Rectified Linear Unit) activation to the tensor.

        Returns:
            A new tensor with ReLU applied
        """
        # Use the ReLU function from autograd
        relu_fn = get_function("relu")
        return relu_fn.forward(self)

    def backward(self, gradient=None):
        """
        Perform backpropagation starting from this tensor.

        Args:
            gradient: Initial gradient value (optional)
        """
        # Ensure proper shape for scalar gradients
        if gradient is None:
            if np.isscalar(self.data) or self.data.size == 1:
                gradient = np.array(1.0)
            else:
                gradient = np.ones_like(self.data)
        elif np.isscalar(gradient):
            gradient = np.array(gradient)

        # Initialize gradient
        self.grad = gradient

        # Build topological ordering of the graph
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    if child.requires_grad:
                        build_topo(child)
                topo.append(node)

        build_topo(self)

        # Backpropagate through the graph
        for node in reversed(topo):
            node._backward()

    def clip_gradients(self, max_norm):
        """
        Clip gradients to prevent exploding gradients.

        Args:
            max_norm: Maximum gradient norm
        """
        if self.grad is None:
            return

        # Calculate gradient norm
        grad_norm = np.sqrt(np.sum(np.square(self.grad)))

        # Clip if necessary
        if grad_norm > max_norm:
            self.grad = self.grad * (max_norm / (grad_norm + 1e-12))

    def zero_grad(self):
        """Reset the gradient to None."""
        self.grad = None