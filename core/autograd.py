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
            self.grad = (
                np.ones_like(self.data) if hasattr(self, "data") else np.array(1.0)
            )
        else:
            self.grad = gradient

        # Backpropagate through the graph in reverse topological order
        for node in reversed(topo_order):
            node.backward_fn()


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
        from core.tensor import Tensor

        # Determine if output requires gradients
        requires_grad = any(t.requires_grad for t in inputs if isinstance(t, Tensor))

        # Create context for storing data needed in backward
        ctx: Dict[str, Any] = {}

        # Convert tensor inputs to numpy arrays for computation
        numpy_inputs = []
        for inp in inputs:
            if isinstance(inp, Tensor):
                numpy_inputs.append(inp.data)
            else:
                numpy_inputs.append(inp)

        # Apply the operation
        output_data = cls.apply(ctx, *numpy_inputs)

        # Create output tensor
        output = Tensor(output_data, requires_grad=requires_grad)

        if requires_grad:
            # Store references to input tensors
            tensor_inputs = [inp for inp in inputs if isinstance(inp, Tensor)]
            output.parents = set(tensor_inputs)

            # Define backward function
            def backward_fn() -> None:
                if output.grad is None:
                    return

                # Compute input gradients
                input_grads = cls.backward(ctx, output.grad)

                # Update gradients of input tensors
                for tensor_input, input_grad in zip(tensor_inputs, input_grads):
                    if tensor_input.requires_grad and input_grad is not None:
                        # Ensure gradients are properly shaped
                        if input_grad.shape != tensor_input.data.shape:
                            # Try to handle broadcasting
                            if tensor_input.data.ndim == input_grad.ndim:
                                # Sum along the broadcasted dimensions
                                axes = tuple(
                                    i
                                    for i, (a, b) in enumerate(
                                        zip(tensor_input.data.shape, input_grad.shape)
                                    )
                                    if a == 1 and b > 1
                                )
                                input_grad = np.sum(
                                    input_grad, axis=axes, keepdims=True
                                )

                        # Accumulate gradient
                        if tensor_input.grad is None:
                            tensor_input.grad = input_grad
                        else:
                            tensor_input.grad += input_grad

            # Set backward function
            output.backward_fn = backward_fn

        return output


class Add(Function):
    """Element-wise addition function."""

    @staticmethod
    def apply(ctx: Dict[str, Any], a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    @staticmethod
    def backward(
        ctx: Dict[str, Any], grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, grad_output


class Multiply(Function):
    """Element-wise multiplication function."""

    @staticmethod
    def apply(ctx: Dict[str, Any], a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx["a"] = a
        ctx["b"] = b
        return a * b

    @staticmethod
    def backward(
        ctx: Dict[str, Any], grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        a, b = ctx["a"], ctx["b"]
        return grad_output * b, grad_output * a


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
        grad_a = grad_output @ b.T
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


class Exp(Function):
    """Exponential function."""

    @staticmethod
    def apply(ctx: Dict[str, Any], a: np.ndarray) -> np.ndarray:
        result = np.exp(a)
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
        ctx["input"] = a
        return np.log(a)

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


class Mean(Function):
    """Mean reduction function."""

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
        ctx["size"] = a.size if axis is None else a.shape[axis]
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

        # Broadcast gradient to input shape and divide by the size of the reduced dimension
        grad_input = np.broadcast_to(grad_output_reshaped, input_shape) / size

        return grad_input, None, None


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
