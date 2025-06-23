"""
Quick model builders for common architectures.

This module provides simple functions to build common neural network
architectures with sensible defaults, making it easy for beginners
to get started quickly.
"""

import numpy as np
from typing import List, Optional, Union, Literal

from nn.modules.container import Sequential
from nn.modules.linear import Linear
from nn.modules.activation import ReLU, Tanh, Softmax
from nn.modules.normalization import BatchNorm
from nn.init.initializers import he_normal, xavier_normal


def MLP(
    layers: List[int],
    activation: Literal['relu', 'tanh'] = 'relu',
    output_activation: Optional[str] = None,
    dropout: float = 0.0,
    batch_norm: bool = False,
    bias: bool = True,
    init: str = 'he'
) -> Sequential:
    """
    Build a Multi-Layer Perceptron (fully connected network).
    
    Args:
        layers: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
        activation: Activation function for hidden layers ('relu', 'tanh')
        output_activation: Activation for output layer (None, 'softmax')
        dropout: Dropout probability (0.0 = no dropout)
        batch_norm: Whether to use batch normalization
        bias: Whether to use bias in linear layers
        init: Weight initialization ('he', 'xavier')
    
    Returns:
        Sequential model
    
    Examples:
        # Simple classifier
        >>> model = MLP([784, 128, 64, 10])
        
        # With batch norm and dropout
        >>> model = MLP([784, 256, 128, 10], batch_norm=True, dropout=0.2)
        
        # Regression model
        >>> model = MLP([100, 50, 1], activation='tanh')
    """
    if len(layers) < 2:
        raise ValueError("Need at least input and output layer sizes")
    
    # Choose activation function
    if activation == 'relu':
        act_fn = ReLU
    elif activation == 'tanh':
        act_fn = Tanh
    else:
        raise ValueError(f"Unknown activation: {activation}")
    
    # Choose initializer
    if init == 'he':
        initializer = he_normal
    elif init == 'xavier':
        initializer = xavier_normal
    else:
        raise ValueError(f"Unknown initializer: {init}")
    
    # Build layers
    model_layers = []
    
    for i in range(len(layers) - 1):
        in_features = layers[i]
        out_features = layers[i + 1]
        
        # Add linear layer
        linear = Linear(in_features, out_features)
        
        # Initialize weights
        linear.weight.data = initializer((in_features, out_features))
        if bias:
            linear.bias.data = np.zeros(out_features)
        
        model_layers.append(linear)
        
        # Add batch norm (before activation)
        if batch_norm and i < len(layers) - 2:  # Not on output layer
            model_layers.append(BatchNorm(out_features))
        
        # Add activation (not on output layer unless specified)
        if i < len(layers) - 2:
            model_layers.append(act_fn())
            
            # Add dropout after activation
            if dropout > 0.0:
                from nn.modules.activation import Dropout
                model_layers.append(Dropout(dropout))
    
    # Add output activation if specified
    if output_activation == 'softmax':
        model_layers.append(Softmax())
    
    return Sequential(*model_layers)


def Classifier(
    input_size: int,
    num_classes: int,
    hidden_sizes: List[int] = [128, 64],
    activation: str = 'relu',
    dropout: float = 0.1,
    batch_norm: bool = True
) -> Sequential:
    """
    Build a classification model with sensible defaults.
    
    Args:
        input_size: Size of input features
        num_classes: Number of output classes
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization
    
    Returns:
        Sequential model ready for classification
    
    Examples:
        # MNIST classifier
        >>> model = Classifier(784, 10)
        
        # CIFAR-10 classifier
        >>> model = Classifier(3072, 10, hidden_sizes=[512, 256, 128])
    """
    layers = [input_size] + hidden_sizes + [num_classes]
    
    return MLP(
        layers=layers,
        activation=activation,
        output_activation='softmax' if num_classes > 1 else None,
        dropout=dropout,
        batch_norm=batch_norm
    )


def Regressor(
    input_size: int,
    output_size: int = 1,
    hidden_sizes: List[int] = [128, 64],
    activation: str = 'relu',
    dropout: float = 0.1,
    batch_norm: bool = False
) -> Sequential:
    """
    Build a regression model with sensible defaults.
    
    Args:
        input_size: Size of input features
        output_size: Size of output (usually 1)
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization
    
    Returns:
        Sequential model ready for regression
    
    Examples:
        # Simple regressor
        >>> model = Regressor(100, 1)
        
        # Multi-output regressor
        >>> model = Regressor(50, 3, hidden_sizes=[256, 128])
    """
    layers = [input_size] + hidden_sizes + [output_size]
    
    return MLP(
        layers=layers,
        activation=activation,
        output_activation=None,  # No activation on regression output
        dropout=dropout,
        batch_norm=batch_norm
    )


def Autoencoder(
    input_size: int,
    encoding_dims: List[int],
    activation: str = 'relu',
    output_activation: str = 'tanh'
) -> Sequential:
    """
    Build an autoencoder with encoder-decoder structure.
    
    Args:
        input_size: Size of input
        encoding_dims: Dimensions for encoding layers (bottleneck is last)
        activation: Activation function for hidden layers
        output_activation: Activation for output layer
    
    Returns:
        Sequential autoencoder model
    
    Examples:
        # Simple autoencoder
        >>> model = Autoencoder(784, [128, 64, 32])
        
        # Creates: 784 -> 128 -> 64 -> 32 -> 64 -> 128 -> 784
    """
    # Encoder layers
    encoder_layers = [input_size] + encoding_dims
    
    # Decoder layers (reverse of encoder, minus the input)
    decoder_layers = encoding_dims[::-1] + [input_size]
    
    # Combine
    all_layers = encoder_layers + decoder_layers[1:]  # Skip duplicate bottleneck
    
    return MLP(
        layers=all_layers,
        activation=activation,
        output_activation=output_activation
    )


def make_model(
    architecture: str,
    input_size: int,
    output_size: int,
    **kwargs
) -> Sequential:
    """
    Factory function to create models by architecture name.
    
    Args:
        architecture: Architecture name ('mlp', 'classifier', 'regressor', 'autoencoder')
        input_size: Input dimension
        output_size: Output dimension
        **kwargs: Additional arguments for the specific architecture
    
    Returns:
        Sequential model
    
    Examples:
        # Quick classifier
        >>> model = make_model('classifier', 784, 10)
        
        # Custom MLP
        >>> model = make_model('mlp', 100, 1, layers=[100, 50, 25, 1])
    """
    if architecture == 'mlp':
        layers = kwargs.get('layers', [input_size, 128, output_size])
        return MLP(layers, **{k: v for k, v in kwargs.items() if k != 'layers'})
    
    elif architecture == 'classifier':
        return Classifier(input_size, output_size, **kwargs)
    
    elif architecture == 'regressor':
        return Regressor(input_size, output_size, **kwargs)
    
    elif architecture == 'autoencoder':
        encoding_dims = kwargs.get('encoding_dims', [128, 64, 32])
        return Autoencoder(input_size, encoding_dims, **kwargs)
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Convenience functions for common use cases
def mnist_classifier(**kwargs) -> Sequential:
    """Quick MNIST classifier (28x28 -> 10 classes)."""
    return Classifier(784, 10, **kwargs)


def cifar10_classifier(**kwargs) -> Sequential:
    """Quick CIFAR-10 classifier (32x32x3 -> 10 classes)."""
    return Classifier(3072, 10, hidden_sizes=[512, 256, 128], **kwargs)


def binary_classifier(input_size: int, **kwargs) -> Sequential:
    """Quick binary classifier."""
    return Regressor(input_size, 1, **kwargs)  # Single output for binary


def simple_nn(
    input_size: int,
    output_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    **kwargs
) -> Sequential:
    """
    Create a simple neural network with uniform hidden layer sizes.
    
    Args:
        input_size: Input dimension
        output_size: Output dimension  
        hidden_size: Size of each hidden layer
        num_layers: Number of hidden layers
        **kwargs: Additional MLP arguments
    
    Returns:
        Sequential model
    
    Examples:
        # 2-layer network
        >>> model = simple_nn(784, 10, hidden_size=256, num_layers=2)
        # Creates: 784 -> 256 -> 256 -> 10
    """
    layers = [input_size] + [hidden_size] * num_layers + [output_size]
    return MLP(layers, **kwargs)


# Model inspection utilities
def model_summary(model: Sequential, input_shape: tuple = None):
    """
    Print a summary of the model architecture.
    
    Args:
        model: Sequential model to summarize
        input_shape: Shape of input (for parameter counting)
    """
    if hasattr(model, 'summary') and input_shape:
        model.summary(input_shape)
    else:
        print("Model Architecture:")
        print("-" * 40)
        total_params = 0
        
        for i, layer in enumerate(model.layers):
            layer_name = layer.__class__.__name__
            
            if hasattr(layer, 'parameters'):
                layer_params = sum(np.prod(p.data.shape) for p in layer.parameters())
                total_params += layer_params
                print(f"{i+1:2d}. {layer_name:<15} - {layer_params:,} params")
            else:
                print(f"{i+1:2d}. {layer_name:<15} - 0 params")
        
        print("-" * 40)
        print(f"Total parameters: {total_params:,}")


def count_parameters(model: Sequential) -> int:
    """Count total number of trainable parameters."""
    return sum(np.prod(p.data.shape) for p in model.parameters())