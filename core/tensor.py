"""
This module implements the Tensor class for automatic differentiation in the ML framework.
"""

import numpy as np


class Tensor:
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
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __repr__(self):
        """Return string representation of the tensor."""
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """
        Add two tensors or a tensor and a scalar.

        Args:
            other: Another tensor or scalar value

        Returns:
            A new tensor containing the result
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                grad = np.ones_like(self.data) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = np.ones_like(other.data) * out.grad
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        out._prev = set()
        if self.requires_grad:
            out._prev.add(self)
        if other.requires_grad:
            out._prev.add(other)

        return out

    def __mul__(self, other):
        """
        Multiply two tensors or a tensor and a scalar.

        Args:
            other: Another tensor or scalar value

        Returns:
            A new tensor containing the result
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = self.data * out.grad
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        out._prev = set()
        if self.requires_grad:
            out._prev.add(self)
        if other.requires_grad:
            out._prev.add(other)

        return out

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

    def __pow__(self, power):
        """
        Raise this tensor to a power.

        Args:
            power: Exponent value

        Returns:
            A new tensor containing the result
        """
        out = Tensor(self.data**power, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
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
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad @ other.data.T
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = self.data.T @ out.grad
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        out._prev = set()
        if self.requires_grad:
            out._prev.add(self)
        if other.requires_grad:
            out._prev.add(other)

        return out

    def __getitem__(self, idx):
        """
        Get items from the tensor at specified indices.

        Args:
            idx: Index or slice to retrieve

        Returns:
            A new tensor with the selected items
        """
        out = Tensor(self.data[idx], requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad
                self.grad = grad if self.grad is None else self.grad + grad

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
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad
                # Broadcast grad to match input shape
                shape = np.ones_like(self.data).sum(axis=axis, keepdims=keepdims).shape
                grad = np.broadcast_to(grad, shape)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = {self}
        return out

    def mean(self):
        """
        Calculate the mean of all tensor elements.

        Returns:
            A new tensor containing the mean value
        """
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.ones_like(self.data) * out.grad / self.data.size
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = {self}
        return out

    def exp(self):
        """
        Calculate the exponential of all tensor elements.

        Returns:
            A new tensor with exponential values
        """
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.exp(self.data) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = {self}
        return out

    def log(self):
        """
        Calculate the natural logarithm of all tensor elements.

        Returns:
            A new tensor with logarithm values
        """
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = (1 / self.data) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = {self}
        return out

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
        out = Tensor(max_data)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                # Handle multiple max values per row (non-unique max)
                mask = self.data == np.max(self.data, axis=axis, keepdims=True)
                grad[mask] = 1.0  # distribute equally if multiple max
                grad_tensor = Tensor(grad)
                if not keepdims and axis is not None:
                    grad_tensor = grad_tensor.reshape(out.shape)  # shape match
                self.backward(grad_tensor)

        out._backward = _backward
        out._prev = {self}
        return out

    def backward(self):
        """
        Perform backpropagation starting from this tensor.

        Computes gradients for all tensors in the computational graph
        that require gradients.
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        print("Backward on:", self.data)
        print("Initial grad:", self.grad)

        visited = set()
        order = []

        def topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    topo(child)
                order.append(tensor)

        topo(self)
        for tensor in reversed(order):
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)
            tensor._backward()

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
