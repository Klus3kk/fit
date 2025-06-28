"""
Core tensor implementation with automatic differentiation support.
"""

import numpy as np
from typing import Any, List, Optional, Set, Tuple, Union

from fit.core.autograd import get_function


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
        """
        Add a tensor or scalar to this tensor.

        Args:
            other: Another tensor or scalar value

        Returns:
            A new tensor containing the result
        """
        # Convert scalar to tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Use the Add function from autograd
        add_fn = get_function("add")
        return add_fn.forward(self, other)

    def __radd__(self, other):
        """Handle right addition (scalar + tensor)."""
        return self.__add__(other)

    def __mul__(self, other):
        """
        Multiply tensor by another tensor or scalar.

        Args:
            other: Another tensor or scalar value

        Returns:
            A new tensor containing the result
        """
        # Convert scalar to tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Use the Multiply function from autograd
        multiply_fn = get_function("multiply")
        return multiply_fn.forward(self, other)

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
        # Convert scalar to tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Use subtraction via addition of negative
        return self + (-1 * other)

    def __rsub__(self, other):
        """Handle right subtraction (scalar - tensor)."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other - self

    def __truediv__(self, other):
        """
        Divide tensor by another tensor or scalar.

        Args:
            other: Another tensor or scalar value

        Returns:
            A new tensor containing the result
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Use multiplication by reciprocal
        return self * (other**-1)

    def __rtruediv__(self, other):
        """Handle right division (scalar / tensor)."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other / self

    def __pow__(self, other):
        """
        Raise tensor to the power of another tensor or scalar.

        Args:
            other: Another tensor or scalar value

        Returns:
            A new tensor containing the result
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Use exp and log: a^b = exp(b * log(a))
        return (other * self.log()).exp()

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
        # Just do it directly with numpy, forget the autograd for now
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
                        grad = np.full_like(self.data, result.grad / self.data.shape[axis])
                    
                    self.grad = grad if self.grad is None else self.grad + grad
            
            result._backward = _backward
            result._prev = {self}
        
        return result

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
            
    @property
    def T(self):
        """Transpose property for 2D tensors with proper gradient support."""
        if len(self.data.shape) != 2:
            raise ValueError("T property only works for 2D tensors")
        
        # Create transposed tensor
        out = Tensor(self.data.T, requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward():
                if out.grad is not None:
                    # print(f"Transpose backward: out.grad.shape = {out.grad.shape}, original.shape = {self.data.shape}")

                    grad = out.grad.T  # Transpose the gradient back to original shape
                    
                    # print(f"  transposed grad.shape = {grad.shape}")
                    self.grad = grad if self.grad is None else self.grad + grad
            
            out._backward = _backward
            out._prev = {self}
        
        return out

    def max(self, axis=None, keepdims=False):
        """
        Find maximum value along specified axis.

        Args:
            axis: Axis along which to find maximum
            keepdims: Whether to keep reduced dimensions

        Returns:
            A new tensor containing the maximum value
        """
        result_data = np.max(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data, requires_grad=self.requires_grad)

        if self.requires_grad:
            # Store which elements are max for gradient computation
            if axis is None:
                # Global maximum
                max_mask = (self.data == np.max(self.data)).astype(np.float64)
            else:
                # Maximum along axis
                max_vals = np.max(self.data, axis=axis, keepdims=True)
                max_mask = (self.data == max_vals).astype(np.float64)

            def _backward():
                if result.grad is not None:
                    # Only the maximum elements get the gradient
                    grad = max_mask * result.grad
                    if axis is not None and not keepdims:
                        # Restore original shape if axis was reduced
                        grad = np.expand_dims(grad, axis=axis)
                    
                    self.grad = grad if self.grad is None else self.grad + grad

            result._backward = _backward
            result._prev = {self}

        return result