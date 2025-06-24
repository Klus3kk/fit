import numpy as np


class DataLoader:
    """Iterable data loader similar to PyTorch's DataLoader."""

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        Initialize a DataLoader.

        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        """Create iterator for the dataset."""
        return DataLoaderIterator(self)

    def __len__(self):
        """Calculate the number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DataLoaderIterator:
    """Iterator for the DataLoader."""

    def __init__(self, dataloader):
        """Initialize the iterator."""
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self.drop_last = dataloader.drop_last

        # Set up indices
        self.indices = np.arange(len(self.dataset))
        if self.dataloader.shuffle:
            np.random.shuffle(self.indices)

        self.current_idx = 0

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self):
        """Get the next batch."""
        from core.tensor import Tensor

        # Check if iteration is complete
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        # Calculate batch size (might be smaller for last batch)
        batch_size = min(self.batch_size, len(self.dataset) - self.current_idx)

        # If we should drop the last batch and it's smaller than batch_size
        if self.drop_last and batch_size < self.batch_size:
            raise StopIteration

        # Get batch indices
        batch_indices = self.indices[self.current_idx : self.current_idx + batch_size]

        # Get batch data
        batch = [self.dataset[i] for i in batch_indices]

        # Update index
        self.current_idx += batch_size

        # Handle different return formats based on dataset
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            # Dataset returns (x, y) pairs
            x_batch = [item[0] for item in batch]
            y_batch = [item[1] for item in batch]

            # Convert to numpy arrays
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            # Convert to Tensors
            x_tensor = Tensor(x_batch, requires_grad=True)
            y_tensor = Tensor(y_batch, requires_grad=False)

            return x_tensor, y_tensor
        else:
            # Dataset returns x only
            x_batch = np.array(batch)
            x_tensor = Tensor(x_batch, requires_grad=True)
            return x_tensor
