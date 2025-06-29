"""
Dataset classes for handling data.
"""

import numpy as np
from typing import Union, Tuple, Any, Optional, Callable, List
from fit.core.tensor import Tensor


class Dataset:
    """
    Base dataset class for handling data.

    Wraps arrays and provides indexing functionality.
    """

    def __init__(
        self,
        X: Union[np.ndarray, Tensor],
        y: Union[np.ndarray, Tensor] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            X: Input features
            y: Target labels (optional for unsupervised tasks)
            transform: Optional transform to apply to features
            target_transform: Optional transform to apply to targets
        """
        # Convert to numpy arrays if needed
        if isinstance(X, Tensor):
            self.X = X.data
        else:
            self.X = np.array(X)

        if y is not None:
            if isinstance(y, Tensor):
                self.y = y.data
            else:
                self.y = np.array(y)
        else:
            self.y = None

        self.transform = transform
        self.target_transform = target_transform

        # Validate shapes
        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError(
                f"X and y must have same length: {len(self.X)} vs {len(self.y)}"
            )

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Get item by index.

        Args:
            idx: Index of the item

        Returns:
            (X, y) tuple if y is provided, else just X
        """
        x = self.X[idx]

        # Apply transform if provided
        if self.transform is not None:
            x = self.transform(x)

        # Convert to tensor
        if not isinstance(x, Tensor):
            x = Tensor(x)

        if self.y is not None:
            y = self.y[idx]

            # Apply target transform if provided
            if self.target_transform is not None:
                y = self.target_transform(y)

            # Convert to tensor
            if not isinstance(y, Tensor):
                y = Tensor(y)

            return x, y
        else:
            return x

    def split(
        self, test_size: float = 0.2, random_state: Optional[int] = None
    ) -> Tuple["Dataset", "Dataset"]:
        """
        Split dataset into train and test sets.

        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            (train_dataset, test_dataset) tuple
        """
        if random_state is not None:
            np.random.seed(random_state)

        n_samples = len(self.X)
        n_test = int(n_samples * test_size)

        # Create random indices
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        # Create train dataset
        X_train = self.X[train_indices]
        y_train = self.y[train_indices] if self.y is not None else None
        train_dataset = Dataset(X_train, y_train, self.transform, self.target_transform)

        # Create test dataset
        X_test = self.X[test_indices]
        y_test = self.y[test_indices] if self.y is not None else None
        test_dataset = Dataset(X_test, y_test, self.transform, self.target_transform)

        return train_dataset, test_dataset

    def shuffle(self, random_state: Optional[int] = None):
        """
        Shuffle the dataset in place.

        Args:
            random_state: Random seed for reproducibility
        """
        if random_state is not None:
            np.random.seed(random_state)

        indices = np.random.permutation(len(self.X))
        self.X = self.X[indices]
        if self.y is not None:
            self.y = self.y[indices]

    def get_subset(self, indices: Union[List[int], np.ndarray]) -> "Dataset":
        """
        Get a subset of the dataset.

        Args:
            indices: Indices to include in subset

        Returns:
            New dataset with selected indices
        """
        X_subset = self.X[indices]
        y_subset = self.y[indices] if self.y is not None else None
        return Dataset(X_subset, y_subset, self.transform, self.target_transform)


class TensorDataset(Dataset):
    """
    Dataset from tensors.

    Specialized dataset for when data is already in tensor format.
    """

    def __init__(self, *tensors: Tensor):
        """
        Initialize tensor dataset.

        Args:
            *tensors: Variable number of tensors (features, targets, etc.)
        """
        if len(tensors) == 0:
            raise ValueError("At least one tensor must be provided")

        # Check that all tensors have the same length
        length = len(tensors[0].data)
        for tensor in tensors[1:]:
            if len(tensor.data) != length:
                raise ValueError("All tensors must have the same length")

        self.tensors = tensors

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.tensors[0].data)

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Get item by index.

        Args:
            idx: Index of the item

        Returns:
            Tensor or tuple of tensors
        """
        items = tuple(Tensor(tensor.data[idx]) for tensor in self.tensors)

        if len(items) == 1:
            return items[0]
        else:
            return items


class ConcatDataset(Dataset):
    """
    Dataset for concatenating multiple datasets.
    """

    def __init__(self, datasets: List[Dataset]):
        """
        Initialize concatenated dataset.

        Args:
            datasets: List of datasets to concatenate
        """
        if len(datasets) == 0:
            raise ValueError("At least one dataset must be provided")

        self.datasets = datasets
        self.cumulative_sizes = self._get_cumulative_sizes()

    def _get_cumulative_sizes(self) -> List[int]:
        """Get cumulative sizes for indexing."""
        cumulative_sizes = []
        cumsum = 0
        for dataset in self.datasets:
            cumsum += len(dataset)
            cumulative_sizes.append(cumsum)
        return cumulative_sizes

    def __len__(self) -> int:
        """Return total size of concatenated datasets."""
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int):
        """
        Get item by global index.

        Args:
            idx: Global index across all datasets

        Returns:
            Item from appropriate dataset
        """
        if idx < 0:
            idx = len(self) + idx

        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # Find which dataset the index belongs to
        dataset_idx = 0
        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                dataset_idx = i
                break

        # Calculate local index within the dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][local_idx]


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.
    """

    def __init__(self, dataset: Dataset, indices: Union[List[int], np.ndarray]):
        """
        Initialize subset.

        Args:
            dataset: Original dataset
            indices: Indices to include in subset
        """
        self.dataset = dataset
        self.indices = np.array(indices)

    def __len__(self) -> int:
        """Return size of subset."""
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Get item by subset index.

        Args:
            idx: Index within subset

        Returns:
            Item from original dataset
        """
        if idx < 0:
            idx = len(self) + idx

        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        return self.dataset[self.indices[idx]]


class RandomSampler:
    """
    Random sampler for datasets.
    """

    def __init__(
        self,
        dataset: Dataset,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize random sampler.

        Args:
            dataset: Dataset to sample from
            replacement: Whether to sample with replacement
            num_samples: Number of samples to draw (default: len(dataset))
            random_state: Random seed
        """
        self.dataset = dataset
        self.replacement = replacement
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.random_state = random_state

    def __iter__(self):
        """Iterator over random indices."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.replacement:
            indices = np.random.choice(
                len(self.dataset), self.num_samples, replace=True
            )
        else:
            indices = np.random.permutation(len(self.dataset))[: self.num_samples]

        return iter(indices)

    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples


class SequentialSampler:
    """
    Sequential sampler for datasets.
    """

    def __init__(self, dataset: Dataset):
        """
        Initialize sequential sampler.

        Args:
            dataset: Dataset to sample from
        """
        self.dataset = dataset

    def __iter__(self):
        """Iterator over sequential indices."""
        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.dataset)
