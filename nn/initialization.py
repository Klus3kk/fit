"""
Weight initialization strategies for neural networks.

This module provides various initialization methods to help with training
deep neural networks. Proper initialization is crucial for gradient flow
and convergence.
"""

import numpy as np
from typing import Tuple, Union, Callable


def xavier_uniform(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """
    Xavier/Glorot uniform initialization.

    Initializes weights with values drawn from a uniform distribution bounded by:
    [-a, a] where a = gain * sqrt(6 / (fan_in + fan_out))

    This helps maintain the variance of activations and gradients across layers,
    which helps with training deeper networks.

    Args:
        shape: Shape of the tensor to initialize
        gain: Scaling factor (default: 1.0)

    Returns:
        Initialized numpy array
    """
    fan_in, fan_out = _calculate_fan_in_fan_out(shape)
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


def xavier_normal(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """
    Xavier/Glorot normal initialization.

    Initializes weights with values drawn from a normal distribution with:
    mean = 0 and std = gain * sqrt(2 / (fan_in + fan_out))

    Args:
        shape: Shape of the tensor to initialize
        gain: Scaling factor (default: 1.0)

    Returns:
        Initialized numpy array
    """
    fan_in, fan_out = _calculate_fan_in_fan_out(shape)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0.0, std, size=shape)


def he_uniform(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """
    He/Kaiming uniform initialization.

    Specifically designed for ReLU activations. Initializes weights from a uniform
    distribution bounded by:
    [-a, a] where a = gain * sqrt(6 / fan_in)

    Args:
        shape: Shape of the tensor to initialize
        gain: Scaling factor (default: 1.0)

    Returns:
        Initialized numpy array
    """
    fan_in, _ = _calculate_fan_in_fan_out(shape)
    limit = gain * np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)


def he_normal(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """
    He/Kaiming normal initialization.

    Specifically designed for ReLU activations. Initializes weights from a normal
    distribution with:
    mean = 0 and std = gain * sqrt(2 / fan_in)

    Args:
        shape: Shape of the tensor to initialize
        gain: Scaling factor (default: 1.0)

    Returns:
        Initialized numpy array
    """
    fan_in, _ = _calculate_fan_in_fan_out(shape)
    std = gain * np.sqrt(2.0 / fan_in)
    return np.random.normal(0.0, std, size=shape)


def lecun_uniform(shape: Tuple[int, ...]) -> np.ndarray:
    """
    LeCun uniform initialization.

    Suitable for Sigmoid activations. Initializes weights from a uniform
    distribution bounded by:
    [-a, a] where a = sqrt(3 / fan_in)

    Args:
        shape: Shape of the tensor to initialize

    Returns:
        Initialized numpy array
    """
    fan_in, _ = _calculate_fan_in_fan_out(shape)
    limit = np.sqrt(3.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)


def lecun_normal(shape: Tuple[int, ...]) -> np.ndarray:
    """
    LeCun normal initialization.

    Suitable for Sigmoid/Tanh activations. Initializes weights from a normal
    distribution with:
    mean = 0 and std = sqrt(1 / fan_in)

    Args:
        shape: Shape of the tensor to initialize

    Returns:
        Initialized numpy array
    """
    fan_in, _ = _calculate_fan_in_fan_out(shape)
    std = np.sqrt(1.0 / fan_in)
    return np.random.normal(0.0, std, size=shape)


def orthogonal(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """
    Orthogonal initialization.

    Generates weight matrices with orthogonal columns, which helps maintain
    gradient norms during backpropagation.

    Args:
        shape: Shape of the tensor to initialize
        gain: Scaling factor (default: 1.0)

    Returns:
        Initialized numpy array
    """
    if len(shape) < 2:
        raise ValueError("Orthogonal initialization requires at least 2 dimensions")

    # For non-square matrices, create a larger square first
    flat_shape = (shape[0], np.prod(shape[1:]).astype(int))

    # Create a random matrix
    a = np.random.normal(0.0, 1.0, flat_shape)

    # Get orthogonal matrix via SVD
    u, _, v = np.linalg.svd(a, full_matrices=False)

    # Pick the appropriate matrix and reshape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)

    # Scale the matrix
    return gain * q


def zeros(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Initialize tensor with zeros.

    Often used for biases.

    Args:
        shape: Shape of the tensor to initialize

    Returns:
        Initialized numpy array
    """
    return np.zeros(shape)


def ones(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Initialize tensor with ones.

    Used in special cases.

    Args:
        shape: Shape of the tensor to initialize

    Returns:
        Initialized numpy array
    """
    return np.ones(shape)


def constant(shape: Tuple[int, ...], val: float) -> np.ndarray:
    """
    Initialize tensor with a constant value.

    Args:
        shape: Shape of the tensor to initialize
        val: Constant value

    Returns:
        Initialized numpy array
    """
    return np.full(shape, val)


def truncated_normal(
    shape: Tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    truncate_at: float = 2.0,
) -> np.ndarray:
    """
    Truncated normal initialization.

    Draws samples from a normal distribution truncated at `truncate_at` standard deviations.
    This avoids extreme values that can cause saturation with some activation functions.

    Args:
        shape: Shape of the tensor to initialize
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        truncate_at: Number of standard deviations to truncate at

    Returns:
        Initialized numpy array
    """
    # Generate twice as many samples as needed
    size = 2 * np.prod(shape)
    tmp = np.random.normal(mean, std, size=int(size))

    # Keep only samples within truncation bounds
    valid = np.abs(tmp - mean) <= truncate_at * std

    # If we don't have enough samples, try again with more
    if np.sum(valid) < np.prod(shape):
        return truncated_normal(shape, mean, std, truncate_at)

    # Take the first prod(shape) valid samples and reshape
    result = tmp[valid][: np.prod(shape)].reshape(shape)
    return result


def uniform_scaling(shape: Tuple[int, ...], factor: float = 1.0) -> np.ndarray:
    """
    Uniform scaling initialization.

    Initializes from a uniform distribution scaled by a factor.

    Args:
        shape: Shape of the tensor to initialize
        factor: Scaling factor

    Returns:
        Initialized numpy array
    """
    scale = factor * np.sqrt(3.0 / np.prod(shape))
    return np.random.uniform(-scale, scale, size=shape)


def sparse(
    shape: Tuple[int, ...], sparsity: float = 0.1, std: float = 0.01
) -> np.ndarray:
    """
    Sparse initialization.

    Creates a mostly-zero matrix with some weights initialized randomly.
    Useful for very large layers.

    Args:
        shape: Shape of the tensor to initialize
        sparsity: Fraction of weights that should be non-zero
        std: Standard deviation for the non-zero weights

    Returns:
        Initialized numpy array
    """
    # Initialize with zeros
    output = np.zeros(shape)

    # Calculate number of non-zero elements
    n_elems = np.prod(shape)
    n_nonzero = int(sparsity * n_elems)

    # Get random indices
    indices = np.random.choice(n_elems, size=n_nonzero, replace=False)

    # Set those indices to random values
    flat_output = output.flatten()
    flat_output[indices] = np.random.normal(0.0, std, size=n_nonzero)

    return flat_output.reshape(shape)


def variance_scaling(
    shape: Tuple[int, ...],
    scale: float = 1.0,
    mode: str = "fan_in",
    distribution: str = "normal",
) -> np.ndarray:
    """
    General variance scaling initialization.

    A versatile initialization method that encompasses Xavier/Glorot, He/Kaiming,
    and other approaches.

    Args:
        shape: Shape of the tensor to initialize
        scale: Scaling factor
        mode: One of "fan_in", "fan_out", or "fan_avg"
        distribution: Either "normal" or "uniform"

    Returns:
        Initialized numpy array
    """
    fan_in, fan_out = _calculate_fan_in_fan_out(shape)

    if mode == "fan_in":
        denominator = fan_in
    elif mode == "fan_out":
        denominator = fan_out
    elif mode == "fan_avg":
        denominator = (fan_in + fan_out) / 2.0
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Expected 'fan_in', 'fan_out', or 'fan_avg'"
        )

    if distribution == "uniform":
        # Uniform distribution [-limit, limit]
        limit = np.sqrt(3.0 * scale / denominator)
        return np.random.uniform(-limit, limit, size=shape)
    elif distribution == "normal":
        # Normal distribution with mean=0 and std=sqrt(scale/denominator)
        std = np.sqrt(scale / denominator)
        return np.random.normal(0.0, std, size=shape)
    else:
        raise ValueError(
            f"Invalid distribution: {distribution}. Expected 'uniform' or 'normal'"
        )


def _calculate_fan_in_fan_out(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Calculate fan_in and fan_out for a weight tensor.

    For a 2D weight matrix, fan_in is the number of input units and
    fan_out is the number of output units.

    For convolutional layers, fan_in is (channels_in * kernel_size) and
    fan_out is (channels_out * kernel_size).

    Args:
        shape: Shape of the tensor

    Returns:
        Tuple of (fan_in, fan_out)
    """
    if len(shape) < 2:
        raise ValueError(
            "Fan in and fan out calculations require at least 2 dimensions"
        )

    if len(shape) == 2:  # Linear layer weights
        fan_in, fan_out = shape
    else:  # Conv or other multi-dimensional weights
        # For conv weights: [out_channels, in_channels, kernel_height, kernel_width]
        receptive_field_size = 1
        for dim in shape[2:]:
            receptive_field_size *= dim

        # First dimension is out_channels (fan_out), second is in_channels (fan_in)
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size

    return fan_in, fan_out


def get_initializer(name: str, **kwargs) -> Callable:
    """
    Get an initializer function by name.

    Args:
        name: Name of the initializer
        **kwargs: Additional arguments to pass to the initializer

    Returns:
        Initializer function that takes a shape and returns a numpy array
    """
    initializers = {
        "xavier_uniform": xavier_uniform,
        "xavier_normal": xavier_normal,
        "he_uniform": he_uniform,
        "he_normal": he_normal,
        "lecun_uniform": lecun_uniform,
        "lecun_normal": lecun_normal,
        "orthogonal": orthogonal,
        "zeros": zeros,
        "ones": ones,
        "constant": constant,
        "truncated_normal": truncated_normal,
        "uniform_scaling": uniform_scaling,
        "sparse": sparse,
        "variance_scaling": variance_scaling,
    }

    if name not in initializers:
        raise ValueError(f"Unknown initializer: {name}")

    initializer = initializers[name]

    def init_fn(shape):
        return initializer(shape, **kwargs)

    return init_fn
