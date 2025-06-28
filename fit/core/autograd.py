"""
This module implements the autograd engine for automatic differentiation.

The autograd engine tracks computations and builds a directed acyclic graph
for efficient backpropagation.
"""

import numpy as np
from typing import Dict, List, Set, Callable, Optional, Tuple, Any, Union

class Node:
    """
    Represents a node in the computational graph.

    Each node represents a value in the computation graph and tracks its
    dependencies (parents) as well as the backward function to compute
    gradients with respect to its inputs.
    """

    def __init__(self, requires_grad: bool = False):
        """
        Initialize a node in the computational graph.

        Args:
            requires_grad: Whether this node requires gradient computation
        """
        self.parents: Set["Node"] = set()
        self.backward_fn: Callable[[], None] = lambda: None
        self.grad: Optional[np.ndarray] = None
        self.requires_grad: bool = requires_grad

    def backward(self, gradient: Optional[np.ndarray] = None) -> None:
        """
        Perform backpropagation starting from this node.

        Args:
            gradient: Upstream gradient to apply (defaults to ones)
        """
        # Build topological ordering of the graph
        topo_order = []
        visited = set()

        def build_topo(node: "Node") -> None:
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    if parent.requires_grad:
                        build_topo(parent)
                topo_order.append(node)

        build_topo(self)

        # Initialize gradient at the start node
        if gradient is None:
            if hasattr(self, "data"):
                # Use ones with the same shape as data
                self.grad = np.ones_like(self.data, dtype=np.float64)
            else:
                # Scalar gradient
                self.grad = np.array(1.0, dtype=np.float64)
        else:
            # Ensure gradient has the correct dtype
            if not isinstance(gradient, np.ndarray):
                gradient = np.array(gradient, dtype=np.float64)
            elif gradient.dtype.kind != "f":
                gradient = gradient.astype(np.float64)

            self.grad = gradient

        # Backpropagate through the graph in reverse topological order
        for node in reversed(topo_order):
            try:
                node.backward_fn()
            except Exception as e:
                # Provide more helpful debugging info if backward fails
                node_info = f"Node type: {type(node).__name__}"
                if hasattr(node, "data"):
                    node_info += f", shape: {node.data.shape}, dtype: {node.data.dtype}"
                print(f"Error in backward for {node_info}: {e}")
                raise


class Function:
    """
    Base class for autograd functions.

    Each function represents an operation in the computation and defines
    how to compute the forward pass and the backward pass (gradient computation).
    """

    @staticmethod
    def apply(ctx: Dict[str, Any], *inputs: Any) -> Any:
        """
        Apply the function to inputs.

        Args:
            ctx: Context dictionary to store data for the backward pass
            *inputs: Input values

        Returns:
            Output value(s)
        """
        raise NotImplementedError("Function subclasses must implement apply")

    @staticmethod
    def backward(
        ctx: Dict[str, Any], grad_output: np.ndarray
    ) -> Tuple[Optional[np.ndarray], ...]:
        """
        Compute gradients with respect to inputs.

        Args:
            ctx: Context dictionary with data stored during the forward pass
            grad_output: Gradient with respect to the output

        Returns:
            Tuple of gradients with respect to inputs
        """
        raise NotImplementedError("Function subclasses must implement backward")

    @classmethod
    def forward(cls, *inputs: "Tensor") -> "Tensor":
        """
        Forward pass of the function.

        Args:
            *inputs: Input tensors

        Returns:
            Output tensor
        """
        from fit.core.tensor import Tensor

        # Determine if output requires gradients
        requires_grad = any(t.requires_grad for t in inputs if isinstance(t, Tensor))

        # Create context for storing data needed in backward
        ctx: Dict[str, Any] = {}

        # Convert tensor inputs to numpy arrays for computation
        numpy_inputs = []
        for inp in inputs:
            # Check if it's a Tensor by looking for the data attribute
            if hasattr(inp, "data") and hasattr(inp, "requires_grad"):
                numpy_inputs.append(inp.data)
            else:
                # Convert non-ndarray inputs to ndarrays
                if not isinstance(inp, np.ndarray):
                    if hasattr(inp, "data"):  # Still a tensor somehow
                        inp = inp.data
                    inp = np.array(inp, dtype=np.float64)
                numpy_inputs.append(inp)

        # Apply the operation
        output_data = cls.apply(ctx, *numpy_inputs)

        # Create output tensor
        output = Tensor(output_data, requires_grad=requires_grad)

        if requires_grad:
            # Store references to input tensors with requires_grad=True
            tensor_inputs = [
                inp for inp in inputs if isinstance(inp, Tensor) and inp.requires_grad
            ]
            output._prev = set(tensor_inputs)
            
            # Define backward function
            def backward_fn():
                if output.grad is not None:
                    grads = cls.backward(ctx, output.grad)
                    for i, (inp, grad) in enumerate(zip(tensor_inputs, grads)):
                        if grad is not None and inp.requires_grad:
                            if inp.grad is None:
                                inp.grad = grad
                            else:
                                inp.grad = inp.grad + grad

            output._backward = backward_fn

        return output


# Helper function for handling broadcasting in gradients
def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Unbroadcast a gradient to match the original tensor shape.

    Args:
        grad: Gradient that may have been broadcast
        shape: Original shape to unbroadcast to

    Returns:
        Unbroadcast gradient that matches the original shape
    """
    # If shapes match, no need to unbroadcast
    if grad.shape == shape:
        return grad

    # For dimensions that were added in broadcasting, sum over them
    grad_ndim = grad.ndim
    shape_ndim = len(shape)
    if grad_ndim > shape_ndim:
        # Sum over added dimensions
        for _ in range(grad_ndim - shape_ndim):
            grad = np.sum(grad, axis=0)

    # For dimensions that were broadcast, sum over the broadcast dimension
    for i, (original_dim, grad_dim) in enumerate(zip(shape, grad.shape)):
        if original_dim == 1 and grad_dim > 1:
            grad = np.sum(grad, axis=i, keepdims=True)

    return grad


# Core autograd functions
class Add(Function):
    """Addition function."""

    @staticmethod
    def apply(ctx: Dict[str, Any], a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx["a_shape"] = a.shape
        ctx["b_shape"] = b.shape
        return a + b

    @staticmethod
    def backward(
        ctx: Dict[str, Any], grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        a_shape, b_shape = ctx["a_shape"], ctx["b_shape"]

        grad_a = grad_output
        grad_b = grad_output

        # Handle broadcasting if needed
        if a_shape != grad_output.shape:
            grad_a = _unbroadcast(grad_a, a_shape)

        if b_shape != grad_output.shape:
            grad_b = _unbroadcast(grad_b, b_shape)

        return grad_a, grad_b


class Multiply(Function):
    """Element-wise multiplication function."""

    @staticmethod
    def apply(ctx: Dict[str, Any], a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx["a"] = a
        ctx["b"] = b
        ctx["a_shape"] = a.shape
        ctx["b_shape"] = b.shape
        return a * b

    @staticmethod
    def backward(
        ctx: Dict[str, Any], grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        a, b = ctx["a"], ctx["b"]
        a_shape, b_shape = ctx["a_shape"], ctx["b_shape"]

        # Compute raw gradients first
        grad_a = grad_output * b
        grad_b = grad_output * a

        # Handle broadcasting if needed
        if a_shape != grad_output.shape:
            grad_a = _unbroadcast(grad_a, a_shape)

        if b_shape != grad_output.shape:
            grad_b = _unbroadcast(grad_b, b_shape)

        return grad_a, grad_b


class MatMul(Function):
    """Matrix multiplication function."""

    @staticmethod
    def apply(ctx: Dict[str, Any], a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx["a"] = a
        ctx["b"] = b
        return a @ b

    @staticmethod
    def backward(
        ctx: Dict[str, Any], grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        a, b = ctx["a"], ctx["b"]

        # Gradient for a: grad_output @ b.T
        # Handle different dimensions correctly
        if grad_output.ndim == 1 and b.ndim == 2:
            # Vector @ matrix case
            grad_a = grad_output.reshape(1, -1) @ b.T
            if a.ndim == 1:
                grad_a = grad_a.reshape(-1)
        else:
            grad_a = grad_output @ b.T

        # Gradient for b: a.T @ grad_output
        # Handle different dimensions correctly
        if a.ndim == 1 and grad_output.ndim > 0:
            # Vector @ matrix case
            a_reshaped = a.reshape(-1, 1)
            grad_b = a_reshaped @ grad_output.reshape(1, -1)
        else:
            grad_b = a.T @ grad_output

        return grad_a, grad_b


class Sum(Function):
    """Sum reduction function."""

    @staticmethod
    def apply(
        ctx: Dict[str, Any],
        a: np.ndarray,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        ctx["input_shape"] = a.shape
        ctx["axis"] = axis
        ctx["keepdims"] = keepdims
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(
        ctx: Dict[str, Any], grad_output: np.ndarray
    ) -> Tuple[np.ndarray, None, None]:
        input_shape = ctx["input_shape"]
        axis = ctx["axis"]
        keepdims = ctx["keepdims"]

        # If keepdims is False, we need to restore dimensions
        if not keepdims and axis is not None:
            # Add back reduced dimensions
            grad_output_reshaped = np.expand_dims(grad_output, axis=axis)
        else:
            grad_output_reshaped = grad_output

        # Broadcast gradient to input shape
        grad_input = np.broadcast_to(grad_output_reshaped, input_shape)

        return grad_input, None, None


class Mean(Function):
    """Mean reduction function."""

    @staticmethod
    def apply(ctx: Dict[str, Any], a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
        ctx["input_shape"] = a.shape
        
        # Handle problematic axis values
        if isinstance(axis, np.ndarray):
            if axis.ndim == 0:  # 0-d array
                axis_val = axis.item()  # Extract the scalar value
                if np.isnan(axis_val):
                    axis = None
                else:
                    axis = int(axis_val)
            else:
                axis = None  # Multi-dimensional axis arrays not supported
        elif axis is not None and np.isnan(axis):
            axis = None
        
        ctx["axis"] = axis
        ctx["keepdims"] = keepdims
        
        if axis is None:
            ctx["size"] = a.size
        else:
            ctx["size"] = a.shape[axis]
            
        return np.mean(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(
        ctx: Dict[str, Any], grad_output: np.ndarray
    ) -> Tuple[np.ndarray, None, None]:
        input_shape = ctx["input_shape"]
        axis = ctx["axis"]
        keepdims = ctx["keepdims"]
        size = ctx["size"]

        # If keepdims is False, we need to restore dimensions
        if not keepdims and axis is not None:
            # Add back reduced dimensions
            grad_output_reshaped = np.expand_dims(grad_output, axis=axis)
        else:
            grad_output_reshaped = grad_output

        # Broadcast gradient to input shape and divide by number of elements
        grad_input = np.broadcast_to(grad_output_reshaped, input_shape) / size

        return grad_input, None, None


class Exp(Function):
    """Exponential function."""

    @staticmethod
    def apply(ctx: Dict[str, Any], a: np.ndarray) -> np.ndarray:
        # To prevent overflow, clip very large values
        a_safe = np.clip(a, -88, 88)  # exp(88) is close to float64 max
        result = np.exp(a_safe)
        ctx["result"] = result
        return result

    @staticmethod
    def backward(ctx: Dict[str, Any], grad_output: np.ndarray) -> Tuple[np.ndarray,]:
        result = ctx["result"]
        return (grad_output * result,)


class Log(Function):
    """Natural logarithm function."""

    @staticmethod
    def apply(ctx: Dict[str, Any], a: np.ndarray) -> np.ndarray:
        # Add small epsilon for numerical stability
        a_safe = np.maximum(a, 1e-12)
        ctx["input"] = a_safe
        return np.log(a_safe)

    @staticmethod
    def backward(ctx: Dict[str, Any], grad_output: np.ndarray) -> Tuple[np.ndarray,]:
        input_data = ctx["input"]
        return (grad_output / input_data,)


class Reshape(Function):
    """Reshape function."""

    @staticmethod
    def apply(ctx: Dict[str, Any], a: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        ctx["input_shape"] = a.shape
        return np.reshape(a, shape)

    @staticmethod
    def backward(
        ctx: Dict[str, Any], grad_output: np.ndarray
    ) -> Tuple[np.ndarray, None]:
        input_shape = ctx["input_shape"]
        return np.reshape(grad_output, input_shape), None


class ReLU(Function):
    """Rectified Linear Unit function."""

    @staticmethod
    def apply(ctx: Dict[str, Any], a: np.ndarray) -> np.ndarray:
        ctx["mask"] = a > 0
        return np.maximum(a, 0)

    @staticmethod
    def backward(ctx: Dict[str, Any], grad_output: np.ndarray) -> Tuple[np.ndarray,]:
        mask = ctx["mask"]
        return (grad_output * mask,)


class Tanh(Function):
    """Hyperbolic tangent function."""

    @staticmethod
    def apply(ctx: Dict[str, Any], a: np.ndarray) -> np.ndarray:
        result = np.tanh(a)
        ctx["result"] = result
        return result

    @staticmethod
    def backward(ctx: Dict[str, Any], grad_output: np.ndarray) -> Tuple[np.ndarray,]:
        result = ctx["result"]
        # Derivative of tanh is 1 - tanh^2
        return (grad_output * (1 - result * result),)


# Register functions for use with the tensor class
function_registry = {
    "add": Add,
    "multiply": Multiply,
    "matmul": MatMul,
    "sum": Sum,
    "exp": Exp,
    "log": Log,
    "reshape": Reshape,
    "relu": ReLU,
    "mean": Mean,
    "tanh": Tanh,
}


# Function to get a registered function
def get_function(name: str) -> Function:
    """
    Get a registered autograd function by name.

    Args:
        name: Name of the function

    Returns:
        Function class
    """
    if name not in function_registry:
        raise ValueError(f"Function {name} not found in registry")

    return function_registry[name]
