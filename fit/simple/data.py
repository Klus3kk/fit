"""
Quick dataset loaders for common machine learning datasets.

This module provides simple functions to load and preprocess popular
datasets, making it easy to get started with machine learning experiments.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Literal
import os

from fit.data.dataset import Dataset
from fit.data.dataloader import DataLoader


def load_dataset(
    name: str,
    batch_size: int = 32,
    validation_split: float = 0.2,
    test_split: Optional[float] = None,
    shuffle: bool = True,
    normalize: bool = True,
    flatten: bool = False,
    data_dir: str = "./data",
    **kwargs,
) -> Dict[str, DataLoader]:
    """
    Load a dataset by name with automatic preprocessing.

    Args:
        name: Dataset name ('mnist', 'cifar10', 'boston', 'iris', 'xor')
        batch_size: Batch size for DataLoaders
        validation_split: Fraction of training data to use for validation
        test_split: Fraction of data to use for testing (if dataset doesn't have test split)
        shuffle: Whether to shuffle the data
        normalize: Whether to normalize features
        flatten: Whether to flatten image data
        data_dir: Directory to store/cache datasets
        **kwargs: Additional dataset-specific arguments

    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataLoaders

    Examples:
        # Load MNIST
        >>> data = load_dataset('mnist', batch_size=64)
        >>> train_loader = data['train']
        >>> val_loader = data['val']

        # Load with custom split
        >>> data = load_dataset('iris', validation_split=0.3, test_split=0.2)
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load the specific dataset
    if name.lower() == "mnist":
        return _load_mnist(
            data_dir, batch_size, validation_split, shuffle, normalize, flatten
        )
    elif name.lower() == "cifar10":
        return _load_cifar10(
            data_dir, batch_size, validation_split, shuffle, normalize, flatten
        )
    elif name.lower() == "boston":
        return _load_boston(
            batch_size, validation_split, test_split, shuffle, normalize
        )
    elif name.lower() == "iris":
        return _load_iris(batch_size, validation_split, test_split, shuffle, normalize)
    elif name.lower() == "xor":
        return _load_xor(batch_size, shuffle, **kwargs)
    elif name.lower() == "spiral":
        return _load_spiral(batch_size, shuffle, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _load_mnist(data_dir, batch_size, validation_split, shuffle, normalize, flatten):
    """Load MNIST dataset."""
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        raise ImportError(
            "sklearn required for MNIST. Install with: pip install scikit-learn"
        )

    print("Loading MNIST dataset...")

    # Fetch MNIST
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        parser="auto",
        cache=True,
        data_home=data_dir,
        as_frame=False,
    )

    X = mnist.data.astype("float32")
    y = mnist.target.astype("int")

    # Normalize
    if normalize:
        X = X / 255.0

    # Reshape if not flattening
    if not flatten:
        X = X.reshape(-1, 28, 28, 1)

    # Split data (MNIST: first 60k train, last 10k test)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Create validation split from training data
    val_size = int(len(X_train) * validation_split)
    if shuffle:
        indices = np.random.permutation(len(X_train))
        X_train, X_val = X_train[indices[val_size:]], X_train[indices[:val_size]]
        y_train, y_val = y_train[indices[val_size:]], y_train[indices[:val_size]]
    else:
        X_train, X_val = X_train[val_size:], X_train[:val_size]
        y_train, y_val = y_train[val_size:], y_train[:val_size]

    # Create datasets and loaders
    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)
    test_dataset = Dataset(X_test, y_test)

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def _load_cifar10(data_dir, batch_size, validation_split, shuffle, normalize, flatten):
    """Load CIFAR-10 dataset."""
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        raise ImportError(
            "sklearn required for CIFAR-10. Install with: pip install scikit-learn"
        )

    print("Loading CIFAR-10 dataset...")

    # Fetch CIFAR-10
    cifar = fetch_openml(
        "CIFAR_10",
        version=1,
        parser="auto",
        cache=True,
        data_home=data_dir,
        as_frame=False,
    )

    X = cifar.data.astype("float32")
    y = cifar.target.astype("int")

    # Normalize
    if normalize:
        X = X / 255.0

    # Reshape if not flattening (CIFAR-10 is 32x32x3)
    if not flatten:
        X = X.reshape(-1, 32, 32, 3)

    # Create train/val/test splits
    total_size = len(X)
    test_size = int(total_size * 0.2)  # 20% for test
    val_size = int((total_size - test_size) * validation_split)

    if shuffle:
        indices = np.random.permutation(total_size)
        X, y = X[indices], y[indices]

    X_test, y_test = X[:test_size], y[:test_size]
    X_val, y_val = (
        X[test_size : test_size + val_size],
        y[test_size : test_size + val_size],
    )
    X_train, y_train = X[test_size + val_size :], y[test_size + val_size :]

    # Create datasets and loaders
    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)
    test_dataset = Dataset(X_test, y_test)

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def _load_boston(batch_size, validation_split, test_split, shuffle, normalize):
    """Load Boston Housing dataset."""
    try:
        from sklearn.datasets import load_boston
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "sklearn required for Boston dataset. Install with: pip install scikit-learn"
        )

    print("Loading Boston Housing dataset...")

    # Load data
    data = load_boston()
    X = data.data.astype("float32")
    y = data.target.astype("float32").reshape(-1, 1)

    # Normalize features
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Create splits
    return _create_splits(X, y, batch_size, validation_split, test_split, shuffle)


def _load_iris(batch_size, validation_split, test_split, shuffle, normalize):
    """Load Iris dataset."""
    try:
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "sklearn required for Iris dataset. Install with: pip install scikit-learn"
        )

    print("Loading Iris dataset...")

    # Load data
    data = load_iris()
    X = data.data.astype("float32")
    y = data.target.astype("int")

    # Normalize features
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Create splits
    return _create_splits(X, y, batch_size, validation_split, test_split, shuffle)


def _load_xor(batch_size, shuffle, noise=0.0, n_samples=1000):
    """Generate XOR dataset."""
    print("Generating XOR dataset...")

    # Generate XOR data
    if n_samples <= 4:
        # Use exact XOR
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
        y = np.array([0, 1, 1, 0], dtype="int")
    else:
        # Generate random XOR data
        X = np.random.rand(n_samples, 2).astype("float32")
        X = np.round(X)  # Make binary
        y = (X[:, 0] != X[:, 1]).astype("int")  # XOR logic

        # Add noise if specified
        if noise > 0:
            X += np.random.normal(0, noise, X.shape).astype("float32")

    # Create single dataset (no splits for XOR)
    dataset = Dataset(X, y)

    return {
        "train": DataLoader(dataset, batch_size=batch_size, shuffle=shuffle),
        "val": DataLoader(dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(dataset, batch_size=batch_size, shuffle=False),
    }


def _load_spiral(batch_size, shuffle, n_samples=1000, noise=0.1, n_spirals=2):
    """Generate spiral dataset for classification."""
    print("Generating spiral dataset...")

    X = []
    y = []

    for i in range(n_spirals):
        r = np.linspace(0.1, 1, n_samples // n_spirals)
        theta = np.linspace(
            i * 2 * np.pi / n_spirals,
            i * 2 * np.pi / n_spirals + 2 * np.pi,
            n_samples // n_spirals,
        )

        x1 = r * np.cos(theta) + np.random.normal(0, noise, len(r))
        x2 = r * np.sin(theta) + np.random.normal(0, noise, len(r))

        X.append(np.column_stack([x1, x2]))
        y.append(np.full(len(r), i))

    X = np.vstack(X).astype("float32")
    y = np.hstack(y).astype("int")

    # Create splits
    return _create_splits(X, y, batch_size, 0.2, 0.2, shuffle)


def _create_splits(X, y, batch_size, validation_split, test_split, shuffle):
    """Create train/val/test splits from data."""
    total_size = len(X)

    # Calculate split sizes
    if test_split:
        test_size = int(total_size * test_split)
        remaining_size = total_size - test_size
        val_size = int(remaining_size * validation_split)
    else:
        test_size = 0
        val_size = int(total_size * validation_split)

    # Shuffle if requested
    if shuffle:
        indices = np.random.permutation(total_size)
        X, y = X[indices], y[indices]

    # Create splits
    if test_split:
        X_test, y_test = X[:test_size], y[:test_size]
        X_val, y_val = (
            X[test_size : test_size + val_size],
            y[test_size : test_size + val_size],
        )
        X_train, y_train = X[test_size + val_size :], y[test_size + val_size :]

        test_dataset = Dataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        X_val, y_val = X[:val_size], y[:val_size]
        X_train, y_train = X[val_size:], y[val_size:]
        test_loader = None

    # Create datasets and loaders
    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)

    result = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    }

    if test_loader:
        result["test"] = test_loader

    return result


def quick_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_split: float = 0.2,
    test_split: Optional[float] = None,
    shuffle: bool = True,
    batch_size: int = 32,
) -> Dict[str, DataLoader]:
    """
    Quickly split any dataset into train/val/test loaders.

    Args:
        X: Input features
        y: Target labels
        validation_split: Fraction for validation
        test_split: Fraction for test (optional)
        shuffle: Whether to shuffle data
        batch_size: Batch size for loaders

    Returns:
        Dictionary with DataLoaders

    Examples:
        # Simple train/val split
        >>> loaders = quick_split(X, y, validation_split=0.2)

        # Train/val/test split
        >>> loaders = quick_split(X, y, validation_split=0.2, test_split=0.1)
    """
    return _create_splits(X, y, batch_size, validation_split, test_split, shuffle)


def get_sample_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a small sample of data for quick testing.

    Args:
        dataset_name: Name of dataset

    Returns:
        Tuple of (X, y) arrays

    Examples:
        # Get XOR data for testing
        >>> X, y = get_sample_data('xor')
        >>> print(X.shape, y.shape)  # (4, 2) (4,)
    """
    if dataset_name == "xor":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
        y = np.array([0, 1, 1, 0], dtype="int")
        return X, y

    elif dataset_name == "iris_sample":
        # First few samples from each class
        X = np.array(
            [
                [5.1, 3.5, 1.4, 0.2],  # setosa
                [4.9, 3.0, 1.4, 0.2],  # setosa
                [7.0, 3.2, 4.7, 1.4],  # versicolor
                [6.4, 3.2, 4.5, 1.5],  # versicolor
                [6.3, 3.3, 6.0, 2.5],  # virginica
                [5.8, 2.7, 5.1, 1.9],  # virginica
            ],
            dtype="float32",
        )
        y = np.array([0, 0, 1, 1, 2, 2], dtype="int")
        return X, y

    elif dataset_name == "regression_sample":
        # Simple linear relationship with noise
        np.random.seed(42)
        X = np.random.randn(100, 1).astype("float32")
        y = (2 * X + 1 + 0.1 * np.random.randn(100, 1)).astype("float32")
        return X, y

    else:
        raise ValueError(f"Unknown sample dataset: {dataset_name}")


# Convenience functions for common workflows
def load_for_classification(dataset_name: str, **kwargs):
    """Load a dataset specifically configured for classification."""
    return load_dataset(dataset_name, **kwargs)


def load_for_regression(dataset_name: str, **kwargs):
    """Load a dataset specifically configured for regression."""
    kwargs.setdefault("normalize", True)
    return load_dataset(dataset_name, **kwargs)


def load_tiny(dataset_name: str, max_samples: int = 1000, **kwargs):
    """Load a tiny version of a dataset for quick experimentation."""
    data = load_dataset(dataset_name, **kwargs)

    # Limit the size of each split
    for split_name, loader in data.items():
        if hasattr(loader.dataset, "data"):
            n_samples = min(max_samples, len(loader.dataset.data))
            loader.dataset.data = loader.dataset.data[:n_samples]
            if (
                hasattr(loader.dataset, "targets")
                and loader.dataset.targets is not None
            ):
                loader.dataset.targets = loader.dataset.targets[:n_samples]

    return data
