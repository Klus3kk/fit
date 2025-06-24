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

        # Handle Tensor input by extracting its data
        if isinstance(data, Tensor):
            data = data.data

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
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Use the Multiply function from autograd
        from core.autograd import Multiply

        return Multiply.forward(self, other)

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
        return self * Tensor(-1)

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
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

        if self.requires_grad:

            def _backward():
                if out.grad is not None:
                    grad = out.grad.reshape(self.data.shape)
                    self.grad = grad if self.grad is None else self.grad + grad

            out._backward = _backward
            out._prev = {self}

        return out

    def zero_grad(self):
        """Set the gradient to None."""
        self.grad = None

    def backward(self, gradient=None):
        """
        Compute gradients by traversing the computational graph backward.

        Args:
            gradient: Upstream gradient (defaults to ones if None)
        """
        if gradient is None:
            # For scalar tensors, default gradient is 1.0
            if self.data.ndim == 0:
                gradient = 1.0
            else:
                gradient = np.ones_like(self.data)

        self.grad = gradient

        # Topological sort for backward pass
        visited = set()
        topo = []

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Call backward functions in reverse topological order
        for node in reversed(topo):
            node._backward()
