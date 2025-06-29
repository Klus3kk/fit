"""
DataLoader for batching and iterating over datasets.
"""

import numpy as np
from typing import Optional, Union, Iterator, Callable, Any
from fit.core.tensor import Tensor
from fit.data.dataset import Dataset, RandomSampler, SequentialSampler


class DataLoader:
    """
    DataLoader for batching and iterating over datasets.

    Provides batching, shuffling, and parallel data loading functionality.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Any] = None,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize DataLoader.

        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            sampler: Custom sampler (overrides shuffle)
            drop_last: Whether to drop the last incomplete batch
            collate_fn: Function to collate samples into batches
            random_state: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or self._default_collate
        self.random_state = random_state

        # Set up sampler
        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset, random_state=random_state)
        else:
            self.sampler = SequentialSampler(dataset)

    def __iter__(self) -> Iterator:
        """Return iterator over batches."""
        batch = []
        for idx in self.sampler:
            batch.append(self.dataset[idx])

            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        # Handle remaining samples
        if len(batch) > 0 and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _default_collate(self, batch):
        """
        Default collate function to combine samples into batches.

        Args:
            batch: List of samples from dataset

        Returns:
            Batched tensor(s)
        """
        if len(batch) == 0:
            return None

        # Check if samples are tuples (X, y) or single tensors
        first_sample = batch[0]

        if isinstance(first_sample, tuple):
            # Handle (X, y) pairs
            batch_x = []
            batch_y = []

            for sample in batch:
                x, y = sample
                batch_x.append(x.data if isinstance(x, Tensor) else x)
                batch_y.append(y.data if isinstance(y, Tensor) else y)

            # Stack into batches
            batched_x = Tensor(np.array(batch_x))
            batched_y = Tensor(np.array(batch_y))

            return batched_x, batched_y
        else:
            # Handle single tensors
            batch_data = []
            for sample in batch:
                batch_data.append(sample.data if isinstance(sample, Tensor) else sample)

            return Tensor(np.array(batch_data))


class BatchSampler:
    """
    Sampler that groups indices into batches.
    """

    def __init__(self, sampler, batch_size: int, drop_last: bool = False):
        """
        Initialize batch sampler.

        Args:
            sampler: Base sampler to use
            batch_size: Size of each batch
            drop_last: Whether to drop the last incomplete batch
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        """Iterator over batches of indices."""
        batch = []
        for idx in self.sampler:
            batch.append(idx)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class WeightedRandomSampler:
    """
    Weighted random sampler for handling class imbalance.
    """

    def __init__(
        self,
        weights: Union[list, np.ndarray],
        num_samples: int,
        replacement: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Initialize weighted random sampler.

        Args:
            weights: Weights for each sample
            num_samples: Number of samples to draw
            replacement: Whether to sample with replacement
            random_state: Random seed
        """
        self.weights = np.array(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.random_state = random_state

        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)

    def __iter__(self):
        """Iterator over weighted random indices."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        indices = np.random.choice(
            len(self.weights),
            size=self.num_samples,
            replace=self.replacement,
            p=self.weights,
        )

        return iter(indices)

    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples


class SubsetRandomSampler:
    """
    Random sampler for a subset of indices.
    """

    def __init__(
        self, indices: Union[list, np.ndarray], random_state: Optional[int] = None
    ):
        """
        Initialize subset random sampler.

        Args:
            indices: Subset of indices to sample from
            random_state: Random seed
        """
        self.indices = np.array(indices)
        self.random_state = random_state

    def __iter__(self):
        """Iterator over shuffled subset indices."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        shuffled = np.random.permutation(self.indices)
        return iter(shuffled)

    def __len__(self) -> int:
        """Return number of indices."""
        return len(self.indices)


class DistributedSampler:
    """
    Sampler for distributed training (placeholder implementation).
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Initialize distributed sampler.

        Args:
            dataset: Dataset to sample from
            num_replicas: Number of processes participating in distributed training
            rank: Rank of current process
            shuffle: Whether to shuffle data
            random_state: Random seed
        """
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.random_state = random_state

        # Calculate samples per replica
        self.num_samples = int(np.ceil(len(dataset) / num_replicas))
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        """Iterator over distributed indices."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.shuffle:
            indices = np.random.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset))

        # Add extra samples to make it evenly divisible
        indices = np.concatenate([indices, indices[: self.total_size - len(indices)]])

        # Subsample for this rank
        start_idx = self.rank * self.num_samples
        end_idx = start_idx + self.num_samples
        subset_indices = indices[start_idx:end_idx]

        return iter(subset_indices)

    def __len__(self) -> int:
        """Return number of samples for this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling."""
        if self.random_state is not None:
            np.random.seed(self.random_state + epoch)


def collate_tensors(batch):
    """
    Collate function for tensor data.

    Args:
        batch: List of tensor samples

    Returns:
        Batched tensor
    """
    if len(batch) == 0:
        return None

    # Stack tensors
    batch_data = []
    for item in batch:
        if isinstance(item, Tensor):
            batch_data.append(item.data)
        else:
            batch_data.append(item)

    return Tensor(np.stack(batch_data))


def collate_sequences(batch, pad_value=0):
    """
    Collate function for variable-length sequences.

    Args:
        batch: List of sequence samples
        pad_value: Value to use for padding

    Returns:
        Padded batch tensor
    """
    if len(batch) == 0:
        return None

    # Find maximum length
    max_len = max(len(seq) for seq in batch)

    # Pad sequences
    padded_batch = []
    for seq in batch:
        if isinstance(seq, Tensor):
            seq_data = seq.data
        else:
            seq_data = np.array(seq)

        # Pad to max length
        if len(seq_data) < max_len:
            pad_width = [(0, max_len - len(seq_data))] + [(0, 0)] * (seq_data.ndim - 1)
            padded = np.pad(seq_data, pad_width, constant_values=pad_value)
        else:
            padded = seq_data

        padded_batch.append(padded)

    return Tensor(np.stack(padded_batch))


def pin_memory(tensor):
    """
    Pin tensor memory (placeholder for GPU acceleration).

    Args:
        tensor: Tensor to pin

    Returns:
        Tensor (unchanged in CPU-only implementation)
    """
    # In a full implementation, this would pin memory for faster GPU transfer
    return tensor


class DataLoaderIter:
    """
    Iterator for DataLoader with additional functionality.
    """

    def __init__(self, loader: DataLoader):
        """
        Initialize DataLoader iterator.

        Args:
            loader: DataLoader to iterate over
        """
        self.loader = loader
        self._iterator = iter(loader)
        self._index = 0

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self):
        """Get next batch."""
        try:
            batch = next(self._iterator)
            self._index += 1
            return batch
        except StopIteration:
            self._index = 0
            raise

    def __len__(self):
        """Return number of batches."""
        return len(self.loader)

    @property
    def batch_index(self):
        """Current batch index."""
        return self._index
