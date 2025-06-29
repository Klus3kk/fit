# File: fit/core/tensor.py

"""
Core tensor implementation with automatic differentiation support.
"""

import numpy as np
from typing import Any, List, Optional, Set, Tuple, Union


class Tensor:
    """
    Core tensor class with automatic differentiation capabilities.

    A tensor stores data and tracks operations for gradient computation.
    """

    def __init__(self, data, requires_grad: bool = False):
        """
        Initialize a tensor.

        Args:
            data: The data array (numpy array or compatible)
            requires_grad: Whether to track operations for gradient computation
        """
        # Handle various input types
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        elif isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, (int, float)):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.integer):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.floating):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.requires_grad = requires_grad
        self.grad = None

        # Backward pass variables
        self._backward = lambda: None  # Default: do nothing
        self._prev: Set["Tensor"] = set()  # Set of tensors that led to this one

    def __repr__(self) -> str:
        """String representation of the tensor."""
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_str})"

    def __add__(self, other):
        """Add a tensor or scalar to this tensor."""
        # Convert scalar to tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if out.requires_grad:

            def _backward():
                if self.requires_grad:
                    # Handle broadcasting for self
                    grad_self = out.grad
                    # Sum out added dims
                    ndims_added = grad_self.ndim - self.data.ndim
                    for i in range(ndims_added):
                        grad_self = grad_self.sum(axis=0)
                    # Sum over broadcasted dims
                    for i, (dim, grad_dim) in enumerate(
                        zip(self.data.shape, grad_self.shape)
                    ):
                        if dim == 1 and grad_dim > 1:
                            grad_self = grad_self.sum(axis=i, keepdims=True)

                    self.grad = (
                        grad_self if self.grad is None else self.grad + grad_self
                    )

                if other.requires_grad:
                    # Handle broadcasting for other
                    grad_other = out.grad
                    # Sum out added dims
                    ndims_added = grad_other.ndim - other.data.ndim
                    for i in range(ndims_added):
                        grad_other = grad_other.sum(axis=0)
                    # Sum over broadcasted dims
                    for i, (dim, grad_dim) in enumerate(
                        zip(other.data.shape, grad_other.shape)
                    ):
                        if dim == 1 and grad_dim > 1:
                            grad_other = grad_other.sum(axis=i, keepdims=True)

                    other.grad = (
                        grad_other if other.grad is None else other.grad + grad_other
                    )

            out._backward = _backward
            out._prev = {self, other}

        return out

    def __radd__(self, other):
        """Handle right addition (scalar + tensor)."""
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply tensor by another tensor or scalar."""
        # Convert scalar to tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if out.requires_grad:

            def _backward():
                if self.requires_grad:
                    # Gradient w.r.t. self is other * grad_output
                    grad_self = out.grad * other.data
                    # Handle broadcasting
                    ndims_added = grad_self.ndim - self.data.ndim
                    for i in range(ndims_added):
                        grad_self = grad_self.sum(axis=0)
                    for i, (dim, grad_dim) in enumerate(
                        zip(self.data.shape, grad_self.shape)
                    ):
                        if dim == 1 and grad_dim > 1:
                            grad_self = grad_self.sum(axis=i, keepdims=True)

                    self.grad = (
                        grad_self if self.grad is None else self.grad + grad_self
                    )

                if other.requires_grad:
                    # Gradient w.r.t. other is self * grad_output
                    grad_other = out.grad * self.data
                    # Handle broadcasting
                    ndims_added = grad_other.ndim - other.data.ndim
                    for i in range(ndims_added):
                        grad_other = grad_other.sum(axis=0)
                    for i, (dim, grad_dim) in enumerate(
                        zip(other.data.shape, grad_other.shape)
                    ):
                        if dim == 1 and grad_dim > 1:
                            grad_other = grad_other.sum(axis=i, keepdims=True)

                    other.grad = (
                        grad_other if other.grad is None else other.grad + grad_other
                    )

            out._backward = _backward
            out._prev = {self, other}

        return out

    def __rmul__(self, other):
        """Handle right multiplication (scalar * tensor)."""
        return self.__mul__(other)

    def __sub__(self, other):
        """Subtract another tensor or scalar from this tensor."""
        # Convert scalar to tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if out.requires_grad:

            def _backward():
                if self.requires_grad:
                    # Gradient w.r.t. self is +1 * grad_output
                    grad_self = out.grad
                    # Handle broadcasting
                    ndims_added = grad_self.ndim - self.data.ndim
                    for i in range(ndims_added):
                        grad_self = grad_self.sum(axis=0)
                    for i, (dim, grad_dim) in enumerate(
                        zip(self.data.shape, grad_self.shape)
                    ):
                        if dim == 1 and grad_dim > 1:
                            grad_self = grad_self.sum(axis=i, keepdims=True)

                    self.grad = (
                        grad_self if self.grad is None else self.grad + grad_self
                    )

                if other.requires_grad:
                    # Gradient w.r.t. other is -1 * grad_output
                    grad_other = -out.grad
                    # Handle broadcasting
                    ndims_added = grad_other.ndim - other.data.ndim
                    for i in range(ndims_added):
                        grad_other = grad_other.sum(axis=0)
                    for i, (dim, grad_dim) in enumerate(
                        zip(other.data.shape, grad_other.shape)
                    ):
                        if dim == 1 and grad_dim > 1:
                            grad_other = grad_other.sum(axis=i, keepdims=True)

                    other.grad = (
                        grad_other if other.grad is None else other.grad + grad_other
                    )

            out._backward = _backward
            out._prev = {self, other}

        return out

    def __rsub__(self, other):
        """Handle right subtraction (scalar - tensor)."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.__sub__(self)

    def __truediv__(self, other):
        """Divide tensor by another tensor or scalar."""
        # Convert scalar to tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if out.requires_grad:

            def _backward():
                if self.requires_grad:
                    # Gradient w.r.t. self is 1/other * grad_output
                    grad_self = out.grad / other.data
                    # Handle broadcasting
                    ndims_added = grad_self.ndim - self.data.ndim
                    for i in range(ndims_added):
                        grad_self = grad_self.sum(axis=0)
                    for i, (dim, grad_dim) in enumerate(
                        zip(self.data.shape, grad_self.shape)
                    ):
                        if dim == 1 and grad_dim > 1:
                            grad_self = grad_self.sum(axis=i, keepdims=True)

                    self.grad = (
                        grad_self if self.grad is None else self.grad + grad_self
                    )

                if other.requires_grad:
                    # Gradient w.r.t. other is -self/other^2 * grad_output
                    grad_other = -out.grad * self.data / (other.data**2)
                    # Handle broadcasting
                    ndims_added = grad_other.ndim - other.data.ndim
                    for i in range(ndims_added):
                        grad_other = grad_other.sum(axis=0)
                    for i, (dim, grad_dim) in enumerate(
                        zip(other.data.shape, grad_other.shape)
                    ):
                        if dim == 1 and grad_dim > 1:
                            grad_other = grad_other.sum(axis=i, keepdims=True)

                    other.grad = (
                        grad_other if other.grad is None else other.grad + grad_other
                    )

            out._backward = _backward
            out._prev = {self, other}

        return out

    def __rtruediv__(self, other):
        """Handle right division (scalar / tensor)."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.__truediv__(self)

    def __pow__(self, exponent):
        """Raise tensor to a power."""
        if not isinstance(exponent, (int, float, np.number)):
            raise TypeError("Exponent must be a number")

        out = Tensor(self.data**exponent, requires_grad=self.requires_grad)

        if self.requires_grad:

            def _backward():
                if out.grad is not None:
                    # Gradient of x^n is n * x^(n-1) * grad_output
                    if exponent == 0:
                        grad_self = np.zeros_like(self.data)
                    else:
                        grad_self = out.grad * exponent * (self.data ** (exponent - 1))

                    self.grad = (
                        grad_self if self.grad is None else self.grad + grad_self
                    )

            out._backward = _backward
            out._prev = {self}

        return out

    def __matmul__(self, other):
        """Matrix multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if out.requires_grad:

            def _backward():
                if self.requires_grad:
                    # For C = A @ B, dA = dC @ B.T
                    grad_self = out.grad @ other.data.T
                    self.grad = (
                        grad_self if self.grad is None else self.grad + grad_self
                    )

                if other.requires_grad:
                    # For C = A @ B, dB = A.T @ dC
                    grad_other = self.data.T @ out.grad
                    other.grad = (
                        grad_other if other.grad is None else other.grad + grad_other
                    )

            out._backward = _backward
            out._prev = {self, other}

        return out

    def sum(self, axis=None, keepdims=False):
        """
        Sum tensor elements along specified axis.

        Args:
            axis: Axis along which to sum
            keepdims: Whether to keep the summed dimensions

        Returns:
            A new tensor with summed values
        """
        # Compute the result directly with numpy
        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data, requires_grad=self.requires_grad)

        if self.requires_grad:

            def _backward():
                if result.grad is not None:
                    # Gradient of sum is just ones in the shape of the input
                    if axis is None:
                        # Sum over all elements - gradient is ones
                        grad = np.ones_like(self.data) * result.grad
                    else:
                        # Sum over specific axis - broadcast gradient back
                        if keepdims:
                            grad = np.broadcast_to(result.grad, self.data.shape)
                        else:
                            # Need to expand dims first, then broadcast
                            grad_expanded = np.expand_dims(result.grad, axis=axis)
                            grad = np.broadcast_to(grad_expanded, self.data.shape)

                    self.grad = grad if self.grad is None else self.grad + grad

            result._backward = _backward
            result._prev = {self}

        return result

    def mean(self, axis=None, keepdims=False):
        """Calculate the mean of tensor elements along specified axis."""
        result_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data, requires_grad=self.requires_grad)

        if self.requires_grad:

            def _backward():
                if result.grad is not None:
                    # Gradient of mean is 1/n
                    if axis is None:
                        grad = np.full_like(self.data, result.grad / self.data.size)
                    else:
                        # Handle axis case
                        n = self.data.shape[axis]
                        if keepdims:
                            grad = np.broadcast_to(result.grad / n, self.data.shape)
                        else:
                            grad_expanded = np.expand_dims(result.grad / n, axis=axis)
                            grad = np.broadcast_to(grad_expanded, self.data.shape)

                    self.grad = grad if self.grad is None else self.grad + grad

            result._backward = _backward
            result._prev = {self}

        return result

    def exp(self):
        """Calculate the exponential of all tensor elements."""
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:

            def _backward():
                if out.grad is not None:
                    # Gradient of exp(x) is exp(x) * grad_output
                    grad_self = out.grad * out.data  # out.data is exp(self.data)
                    self.grad = (
                        grad_self if self.grad is None else self.grad + grad_self
                    )

            out._backward = _backward
            out._prev = {self}

        return out

    def log(self):
        """Calculate the natural logarithm of all tensor elements."""
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:

            def _backward():
                if out.grad is not None:
                    # Gradient of log(x) is 1/x * grad_output
                    grad_self = out.grad / self.data
                    self.grad = (
                        grad_self if self.grad is None else self.grad + grad_self
                    )

            out._backward = _backward
            out._prev = {self}

        return out

    def backward(self):
        """
        Perform backpropagation starting from this tensor.
        """
        # Initialize gradient for this tensor if not set
        if self.grad is None:
            if self.data.shape == ():
                # Scalar tensor
                self.grad = np.array(1.0)
            else:
                # Non-scalar tensor - initialize with ones
                self.grad = np.ones_like(self.data)

        # Build topological order of computation graph
        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build_topo(parent)
                topo.append(tensor)

        build_topo(self)

        # Go through the topological order backwards and call backward functions
        for tensor in reversed(topo):
            tensor._backward()

    def __hash__(self):
        """Hash function for tensor objects."""
        return id(self)

    def __eq__(self, other):
        """Check if two tensor objects are the same."""
        return id(self) == id(other)

    @property
    def shape(self):
        """Return the shape of the tensor data."""
        return self.data.shape

    @property
    def ndim(self):
        """Return the number of dimensions."""
        return self.data.ndim

    @property
    def size(self):
        """Return the total number of elements."""
        return self.data.size

    def reshape(self, *shape):
        """Reshape the tensor to the given shape."""
        new_shape = (
            shape[0]
            if len(shape) == 1 and isinstance(shape[0], (tuple, list))
            else shape
        )

        out = Tensor(self.data.reshape(new_shape), requires_grad=self.requires_grad)

        if self.requires_grad:

            def _backward():
                if out.grad is not None:
                    self.grad = (
                        out.grad.reshape(self.data.shape)
                        if self.grad is None
                        else self.grad + out.grad.reshape(self.data.shape)
                    )

            out._backward = _backward
            out._prev = {self}

        return out

    def __getitem__(self, idx):
        """Get items from the tensor at specified indices."""
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

    @property
    def T(self):
        """Return the transpose of the tensor."""
        out = Tensor(self.data.T, requires_grad=self.requires_grad)

        if self.requires_grad:

            def _backward():
                if out.grad is not None:
                    # Gradient of transpose is just transpose of gradient
                    self.grad = (
                        out.grad.T if self.grad is None else self.grad + out.grad.T
                    )

            out._backward = _backward
            out._prev = {self}

        return out
