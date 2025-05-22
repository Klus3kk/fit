"""
This module implements specialized operations for tensors.

These operations build on the core tensor capabilities and autograd engine
to provide higher-level functionality.
"""

import numpy as np
from typing import List, Optional, Tuple, Union

from core.autograd import Function, get_function
from core.tensor import Tensor


# Matrix Operations


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Perform matrix multiplication: a @ b.

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        Result tensor
    """
    return a @ b


def transpose(x: Tensor, axes: Optional[Tuple[int, ...]] = None) -> Tensor:
    """
    Transpose a tensor.

    Args:
        x: Input tensor
        axes: Permutation of dimensions (default is to reverse dimensions)

    Returns:
        Transposed tensor
    """
    if not x.requires_grad:
        return Tensor(np.transpose(x.data, axes))

    # Define operation for transpose
    class Transpose(Function):
        @staticmethod
        def apply(ctx, a, axes=None):
            ctx["axes"] = axes
            return np.transpose(a, axes)

        @staticmethod
        def backward(ctx, grad_output):
            # Transpose the gradient with inverse permutation
            axes = ctx["axes"]
            if axes is None:
                # If axes is None, the transpose reverses the dimensions
                return np.transpose(grad_output), None

            # Get inverse permutation
            n = len(axes)
            inverse_axes = np.zeros(n, dtype=int)
            for i, ax in enumerate(axes):
                inverse_axes[ax] = i

            return np.transpose(grad_output, tuple(inverse_axes)), None

    # Register function temporarily
    function_registry = {}
    function_registry["transpose"] = Transpose

    # Call the function
    result = Transpose.forward(x, axes)

    return result


def einsum(equation: str, *tensors: Tensor) -> Tensor:
    """
    Einstein summation for tensors.

    Args:
        equation: Summation equation in Einstein notation
        *tensors: Input tensors

    Returns:
        Result tensor
    """
    # Check if gradients are needed
    requires_grad = any(t.requires_grad for t in tensors)

    if not requires_grad:
        tensor_data = [t.data for t in tensors]
        return Tensor(np.einsum(equation, *tensor_data))

    # Define operation for einsum
    class Einsum(Function):
        @staticmethod
        def apply(ctx, equation, *inputs):
            ctx["equation"] = equation
            ctx["input_shapes"] = [inp.shape for inp in inputs]
            return np.einsum(equation, *inputs)

        @staticmethod
        def backward(ctx, grad_output):
            equation = ctx["equation"]
            input_shapes = ctx["input_shapes"]

            # Parse the equation
            input_part, output_part = equation.split("->")
            input_subscripts = input_part.split(",")

            grads = []
            for i, subscript in enumerate(input_subscripts):
                # Create gradient equation
                grad_equation = f"{output_part},{','.join([s for j, s in enumerate(input_subscripts) if j != i])}->{''.join(subscript)}"

                # Determine the order of operands for the gradient computation
                operands = [grad_output]
                for j, t in enumerate(inputs):
                    if j != i:
                        operands.append(t)

                # Compute gradient
                grads.append(
                    np.einsum(grad_equation, *operands).reshape(input_shapes[i])
                )

            # Add None for equation parameter
            return (None,) + tuple(grads)

    # Get tensor data
    tensor_data = [t.data for t in tensors]

    # Call the function
    result = Einsum.forward(equation, *tensor_data)

    # Create output tensor
    output = Tensor(result, requires_grad=requires_grad)

    if requires_grad:
        # Store references to input tensors
        output._prev = set(t for t in tensors if t.requires_grad)

        # Define backward function
        def backward_fn():
            if output.grad is None:
                return

            # Compute input gradients
            input_grads = Einsum.backward(
                {"equation": equation, "input_shapes": [t.data.shape for t in tensors]},
                output.grad,
            )

            # Skip the first element (None for equation parameter)
            input_grads = input_grads[1:]

            # Update gradients of input tensors
            for t, input_grad in zip(tensors, input_grads):
                if t.requires_grad and input_grad is not None:
                    t.grad = input_grad if t.grad is None else t.grad + input_grad

        # Set backward function
        output.backward_fn = backward_fn

    return output


# Activation Functions


def sigmoid(x: Tensor) -> Tensor:
    """
    Apply sigmoid activation: 1 / (1 + exp(-x)).

    Args:
        x: Input tensor

    Returns:
        Output tensor
    """

    # Define sigmoid function
    class Sigmoid(Function):
        @staticmethod
        def apply(ctx, a):
            result = 1.0 / (1.0 + np.exp(-a))
            ctx["output"] = result
            return result

        @staticmethod
        def backward(ctx, grad_output):
            output = ctx["output"]
            return (grad_output * output * (1 - output),)

    # Call the function
    return Sigmoid.forward(x)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Apply softmax activation along specified axis.

    Args:
        x: Input tensor
        axis: Axis along which to apply softmax

    Returns:
        Output tensor
    """

    # Define softmax function
    class Softmax(Function):
        @staticmethod
        def apply(ctx, a, axis):
            # For numerical stability, subtract max
            a_max = np.max(a, axis=axis, keepdims=True)
            exp_a = np.exp(a - a_max)
            result = exp_a / np.sum(exp_a, axis=axis, keepdims=True)
            ctx["output"] = result
            ctx["axis"] = axis
            return result

        @staticmethod
        def backward(ctx, grad_output):
            output = ctx["output"]
            axis = ctx["axis"]

            # Compute Jacobian-vector product efficiently
            # For each position i:
            # dL/dy_i = Σ_j dL/dx_j * dx_j/dy_i
            # dx_j/dy_i = y_i * (δ_ij - y_j)

            # Sum(dL/dx_i * y_i)
            S = np.sum(grad_output * output, axis=axis, keepdims=True)

            # dL/dy_i = y_i * (dL/dx_i - S)
            grad_input = output * (grad_output - S)

            return grad_input, None

    # Call the function
    return Softmax.forward(x, axis)


def tanh(x: Tensor) -> Tensor:
    """
    Apply hyperbolic tangent activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).

    Args:
        x: Input tensor

    Returns:
        Output tensor
    """

    # Define tanh function
    class Tanh(Function):
        @staticmethod
        def apply(ctx, a):
            result = np.tanh(a)
            ctx["output"] = result
            return result

        @staticmethod
        def backward(ctx, grad_output):
            output = ctx["output"]
            return (grad_output * (1 - output * output),)

    # Call the function
    return Tanh.forward(x)


# Mathematical Operations


def abs(x: Tensor) -> Tensor:
    """
    Compute the absolute value of a tensor.

    Args:
        x: Input tensor

    Returns:
        Output tensor
    """

    # Define abs function
    class Abs(Function):
        @staticmethod
        def apply(ctx, a):
            ctx["input"] = a
            return np.abs(a)

        @staticmethod
        def backward(ctx, grad_output):
            input_data = ctx["input"]
            # Gradient is sign of input
            return (grad_output * np.sign(input_data),)

    # Call the function
    return Abs.forward(x)


def sqrt(x: Tensor) -> Tensor:
    """
    Compute the square root of a tensor.

    Args:
        x: Input tensor

    Returns:
        Output tensor
    """

    # Define sqrt function
    class Sqrt(Function):
        @staticmethod
        def apply(ctx, a):
            result = np.sqrt(a)
            ctx["output"] = result
            return result

        @staticmethod
        def backward(ctx, grad_output):
            output = ctx["output"]
            return (grad_output / (2 * output),)

    # Call the function
    return Sqrt.forward(x)


def sin(x: Tensor) -> Tensor:
    """
    Compute the sine of a tensor (in radians).

    Args:
        x: Input tensor

    Returns:
        Output tensor
    """

    # Define sin function
    class Sin(Function):
        @staticmethod
        def apply(ctx, a):
            ctx["input"] = a
            return np.sin(a)

        @staticmethod
        def backward(ctx, grad_output):
            input_data = ctx["input"]
            return (grad_output * np.cos(input_data),)

    # Call the function
    return Sin.forward(x)


def cos(x: Tensor) -> Tensor:
    """
    Compute the cosine of a tensor (in radians).

    Args:
        x: Input tensor

    Returns:
        Output tensor
    """

    # Define cos function
    class Cos(Function):
        @staticmethod
        def apply(ctx, a):
            ctx["input"] = a
            return np.cos(a)

        @staticmethod
        def backward(ctx, grad_output):
            input_data = ctx["input"]
            return (grad_output * (-np.sin(input_data)),)

    # Call the function
    return Cos.forward(x)


# Tensor Creation and Manipulation


def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    """
    Create a tensor filled with zeros.

    Args:
        shape: Shape of the tensor
        requires_grad: Whether the tensor requires gradient computation

    Returns:
        Tensor filled with zeros
    """
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    """
    Create a tensor filled with ones.

    Args:
        shape: Shape of the tensor
        requires_grad: Whether the tensor requires gradient computation

    Returns:
        Tensor filled with ones
    """
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def randn(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    """
    Create a tensor filled with random values from a standard normal distribution.

    Args:
        shape: Shape of the tensor
        requires_grad: Whether the tensor requires gradient computation

    Returns:
        Tensor filled with random values
    """
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad)


def concatenate(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """
    Concatenate tensors along specified axis.

    Args:
        tensors: List of tensors to concatenate
        axis: Axis along which to concatenate

    Returns:
        Concatenated tensor
    """
    # Check if gradients are needed
    requires_grad = any(t.requires_grad for t in tensors)

    if not requires_grad:
        tensor_data = [t.data for t in tensors]
        return Tensor(np.concatenate(tensor_data, axis=axis))

    # Define operation for concatenation
    class Concatenate(Function):
        @staticmethod
        def apply(ctx, inputs, axis):
            shapes = [inp.shape for inp in inputs]
            ctx["shapes"] = shapes
            ctx["axis"] = axis
            return np.concatenate(inputs, axis=axis)

        @staticmethod
        def backward(ctx, grad_output):
            shapes = ctx["shapes"]
            axis = ctx["axis"]

            # Compute indices for slicing
            indices = [0]
            for shape in shapes:
                indices.append(indices[-1] + shape[axis])

            # Slice gradient for each input
            grads = []
            for i in range(len(shapes)):
                shape = shapes[i]
                start, end = indices[i], indices[i + 1]

                # Create slice object
                slices = [slice(None)] * grad_output.ndim
                slices[axis] = slice(start, end)

                grads.append(grad_output[tuple(slices)])

            # Add None for concatenation parameters
            return grads + [None]

    # Get tensor data
    tensor_data = [t.data for t in tensors]

    # Create output tensor
    result = Concatenate.forward(tensor_data, axis)
    output = Tensor(result, requires_grad=requires_grad)

    if requires_grad:
        # Store references to input tensors
        output._prev = set(t for t in tensors if t.requires_grad)

        # Define backward function
        def backward_fn():
            if output.grad is None:
                return

            # Compute input gradients
            grads = Concatenate.backward(
                {"shapes": [t.data.shape for t in tensors], "axis": axis}, output.grad
            )

            # Skip the last element (None for axis parameter)
            grads = grads[:-1]

            # Update gradients of input tensors
            for t, grad in zip(tensors, grads):
                if t.requires_grad:
                    t.grad = grad if t.grad is None else t.grad + grad

        # Set backward function
        output.backward_fn = backward_fn

    return output


# Statistical and Reduction Operations


def std(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """
    Compute the standard deviation of a tensor.

    Args:
        x: Input tensor
        axis: Axis along which to compute standard deviation
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Standard deviation tensor
    """
    # Use variance then square root
    return sqrt(var(x, axis, keepdims))


def var(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """
    Compute the variance of a tensor.

    Args:
        x: Input tensor
        axis: Axis along which to compute variance
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Variance tensor
    """

    # Define var function
    class Var(Function):
        @staticmethod
        def apply(ctx, a, axis, keepdims):
            mean = np.mean(a, axis=axis, keepdims=True)
            ctx["input"] = a
            ctx["mean"] = mean
            ctx["axis"] = axis
            ctx["keepdims"] = keepdims

            # Sum of squared deviations
            result = np.mean((a - mean) ** 2, axis=axis, keepdims=keepdims)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            input_data = ctx["input"]
            mean = ctx["mean"]
            axis = ctx["axis"]
            keepdims = ctx["keepdims"]

            # Reshape grad_output for broadcasting if needed
            if not keepdims and axis is not None:
                grad_output = np.expand_dims(grad_output, axis=axis)

            # Compute gradient: 2 * (x - mean) / n
            n = input_data.shape[axis] if axis is not None else input_data.size
            grad_input = 2 * (input_data - mean) * grad_output / n

            return grad_input, None, None

    # Call the function
    return Var.forward(x, axis, keepdims)


def argmax(x: Tensor, axis: Optional[int] = None) -> Tensor:
    """
    Return the indices of the maximum values along specified axis.

    Args:
        x: Input tensor
        axis: Axis along which to find maximum values

    Returns:
        Indices tensor
    """
    # This operation is not differentiable, so we don't need backward
    return Tensor(np.argmax(x.data, axis=axis))


def logsumexp(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """
    Compute the log of the sum of exponentials of input elements.

    This is a numerically stable version of log(sum(exp(x))).

    Args:
        x: Input tensor
        axis: Axis along which to perform the operation
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Result tensor
    """

    # Define logsumexp function
    class LogSumExp(Function):
        @staticmethod
        def apply(ctx, a, axis, keepdims):
            a_max = np.max(a, axis=axis, keepdims=True)
            ctx["a_max"] = a_max
            ctx["input"] = a
            ctx["axis"] = axis
            ctx["keepdims"] = keepdims

            exp_a = np.exp(a - a_max)
            sum_exp = np.sum(exp_a, axis=axis, keepdims=keepdims)
            result = (
                np.log(sum_exp) + a_max.reshape(sum_exp.shape)
                if not keepdims
                else np.log(sum_exp) + a_max
            )

            return result

        @staticmethod
        def backward(ctx, grad_output):
            input_data = ctx["input"]
            a_max = ctx["a_max"]
            axis = ctx["axis"]
            keepdims = ctx["keepdims"]

            # Reshape grad_output for broadcasting if needed
            if not keepdims and axis is not None:
                grad_output = np.expand_dims(grad_output, axis=axis)

            # Compute softmax
            exp_a = np.exp(input_data - a_max)
            sum_exp = np.sum(exp_a, axis=axis, keepdims=True)
            softmax = exp_a / sum_exp

            # Gradient is grad_output * softmax
            grad_input = grad_output * softmax

            return grad_input, None, None

    # Call the function
    return LogSumExp.forward(x, axis, keepdims)


# Loss Functions


def mse_loss(predictions: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """
    Compute Mean Squared Error loss.

    Args:
        predictions: Predicted values
        targets: Target values
        reduction: Reduction method ('mean' or 'sum')

    Returns:
        Loss tensor
    """
    diff = predictions - targets
    squared_diff = diff * diff

    if reduction == "mean":
        return squared_diff.mean()
    elif reduction == "sum":
        return squared_diff.sum()
    else:
        return squared_diff


def binary_cross_entropy(
    predictions: Tensor, targets: Tensor, reduction: str = "mean"
) -> Tensor:
    """
    Compute Binary Cross Entropy loss.

    Args:
        predictions: Predicted probabilities (between 0 and 1)
        targets: Binary target values (0 or 1)
        reduction: Reduction method ('mean' or 'sum')

    Returns:
        Loss tensor
    """
    # Add small epsilon for numerical stability
    epsilon = 1e-12
    predictions = predictions.data.clip(epsilon, 1 - epsilon)

    # Define BCE function
    class BCE(Function):
        @staticmethod
        def apply(ctx, pred, target, reduction):
            ctx["pred"] = pred
            ctx["target"] = target
            ctx["reduction"] = reduction

            # Compute BCE: -t * log(p) - (1 - t) * log(1 - p)
            loss = -target * np.log(pred) - (1 - target) * np.log(1 - pred)

            if reduction == "mean":
                return np.mean(loss)
            elif reduction == "sum":
                return np.sum(loss)
            else:
                return loss

        @staticmethod
        def backward(ctx, grad_output):
            pred = ctx["pred"]
            target = ctx["target"]
            reduction = ctx["reduction"]

            # Gradient: (p - t) / (p * (1 - p))
            grad = (pred - target) / (pred * (1 - pred))

            if reduction == "mean":
                grad = grad * grad_output / pred.size
            elif reduction == "sum":
                grad = grad * grad_output
            else:
                # Reshape grad_output for broadcasting if needed
                if grad_output.ndim < grad.ndim:
                    grad_output = np.expand_dims(
                        grad_output, axis=tuple(range(1, grad.ndim))
                    )
                grad = grad * grad_output

            return grad, None, None

    # Call the function
    return BCE.forward(predictions.data, targets.data, reduction)


def softmax_cross_entropy(
    logits: Tensor, targets: Tensor, reduction: str = "mean"
) -> Tensor:
    """
    Compute Softmax Cross Entropy loss.

    Args:
        logits: Unnormalized predictions (logits)
        targets: Target class indices or one-hot encoded targets
        reduction: Reduction method ('mean' or 'sum')

    Returns:
        Loss tensor
    """

    # Define SCE function
    class SoftmaxCrossEntropy(Function):
        @staticmethod
        def apply(ctx, logits, targets, reduction):
            # For numerical stability
            logits = logits - np.max(logits, axis=1, keepdims=True)
            ctx["logits"] = logits

            # Convert targets to one-hot if they are class indices
            if targets.ndim == 1:
                batch_size, num_classes = logits.shape
                one_hot = np.zeros((batch_size, num_classes))
                one_hot[np.arange(batch_size), targets] = 1
                targets = one_hot

            ctx["targets"] = targets
            ctx["reduction"] = reduction

            # Compute softmax probabilities
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Compute cross entropy: -sum(targets * log(probs))
            log_probs = np.log(probs)
            loss = -np.sum(targets * log_probs, axis=1)

            if reduction == "mean":
                return np.mean(loss)
            elif reduction == "sum":
                return np.sum(loss)
            else:
                return loss

        @staticmethod
        def backward(ctx, grad_output):
            logits = ctx["logits"]
            targets = ctx["targets"]
            reduction = ctx["reduction"]

            # Softmax probabilities
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Gradient: p - t
            batch_size = logits.shape[0]
            grad = probs - targets

            if reduction == "mean":
                grad = grad * grad_output / batch_size
            elif reduction == "sum":
                grad = grad * grad_output
            else:
                # Reshape grad_output for broadcasting
                grad = grad * grad_output.reshape(-1, 1)

            return grad, None, None

    # Call the function
    return SoftmaxCrossEntropy.forward(logits.data, targets.data, reduction)


# Distance and Similarity Functions


def cosine_similarity(
    x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8
) -> Tensor:
    """
    Compute cosine similarity between tensors along specified dimension.

    Args:
        x1: First tensor
        x2: Second tensor
        dim: Dimension along which to compute similarity
        eps: Small value to avoid division by zero

    Returns:
        Similarity tensor
    """

    # Define cosine similarity function
    class CosineSimilarity(Function):
        @staticmethod
        def apply(ctx, a, b, dim, eps):
            ctx["a"] = a
            ctx["b"] = b
            ctx["dim"] = dim
            ctx["eps"] = eps

            # Compute the dot product and norms
            dot_product = np.sum(a * b, axis=dim)
            a_norm = np.sqrt(np.sum(a * a, axis=dim))
            b_norm = np.sqrt(np.sum(b * b, axis=dim))

            # Compute cosine similarity
            similarity = dot_product / np.maximum(a_norm * b_norm, eps)

            # Store intermediate results for backward pass
            ctx["a_norm"] = a_norm
            ctx["b_norm"] = b_norm
            ctx["similarity"] = similarity

            return similarity

        @staticmethod
        def backward(ctx, grad_output):
            a = ctx["a"]
            b = ctx["b"]
            dim = ctx["dim"]
            eps = ctx["eps"]
            a_norm = ctx["a_norm"]
            b_norm = ctx["b_norm"]
            similarity = ctx["similarity"]

            # Reshape for broadcasting
            reshape_dim = lambda x: np.expand_dims(x, dim)
            grad_output_reshaped = reshape_dim(grad_output)
            a_norm_reshaped = reshape_dim(a_norm)
            b_norm_reshaped = reshape_dim(b_norm)
            similarity_reshaped = reshape_dim(similarity)

            # Compute gradients
            ab_norm = np.maximum(a_norm * b_norm, eps)

            # Gradient for a
            grad_a = grad_output_reshaped * (
                b / ab_norm - similarity_reshaped * a / a_norm_reshaped**2
            )

            # Gradient for b
            grad_b = grad_output_reshaped * (
                a / ab_norm - similarity_reshaped * b / b_norm_reshaped**2
            )

            return grad_a, grad_b, None, None

    # Call the function
    return CosineSimilarity.forward(x1.data, x2.data, dim, eps)


def pairwise_distance(
    x1: Tensor, x2: Tensor, p: float = 2.0, eps: float = 1e-6
) -> Tensor:
    """
    Compute pairwise distance between tensors.

    Args:
        x1: First tensor (batch_size x D)
        x2: Second tensor (batch_size x D)
        p: p-norm to use for distance calculation
        eps: Small value to avoid division by zero

    Returns:
        Distance tensor (batch_size)
    """

    # Define pairwise distance function
    class PairwiseDistance(Function):
        @staticmethod
        def apply(ctx, a, b, p, eps):
            ctx["a"] = a
            ctx["b"] = b
            ctx["p"] = p
            ctx["eps"] = eps

            # Compute p-norm
            diff = a - b
            abs_diff = np.abs(diff)

            if p == 1:
                # L1 distance
                return np.sum(abs_diff, axis=1)
            elif p == 2:
                # L2 distance
                return np.sqrt(np.sum(diff * diff, axis=1) + eps)
            else:
                # General p-norm
                p_diff = abs_diff**p
                return np.sum(p_diff, axis=1) ** (1 / p)

        @staticmethod
        def backward(ctx, grad_output):
            a = ctx["a"]
            b = ctx["b"]
            p = ctx["p"]
            eps = ctx["eps"]

            # Compute gradient
            diff = a - b
            abs_diff = np.abs(diff)

            if p == 1:
                # L1 gradient: sign(a - b)
                grad = np.sign(diff)
            elif p == 2:
                # L2 gradient: (a - b) / ||a - b||_2
                norm = np.sqrt(np.sum(diff * diff, axis=1, keepdims=True) + eps)
                grad = diff / norm
            else:
                # General p-norm gradient
                norm = np.sum(abs_diff**p, axis=1, keepdims=True) ** (1 / p - 1)
                grad = p * (abs_diff ** (p - 1)) * np.sign(diff) * norm

            # Multiply by upstream gradient
            grad_output_reshaped = np.expand_dims(grad_output, 1)
            grad_a = grad * grad_output_reshaped
            grad_b = -grad * grad_output_reshaped

            return grad_a, grad_b, None, None

    # Call the function
    return PairwiseDistance.forward(x1.data, x2.data, p, eps)
