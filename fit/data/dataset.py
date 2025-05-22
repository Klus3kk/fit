import numpy as np


class Dataset:
    """Base dataset class similar to PyTorch's Dataset."""

    def __init__(self, data, targets=None, transform=None):
        """
        Initialize a Dataset.

        Args:
            data: Input data
            targets: Target labels/values (optional)
            transform: Function to transform data (optional)
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        x = self.data[idx]

        if self.transform:
            x = self.transform(x)

        if self.targets is not None:
            y = self.targets[idx]
            return x, y
        else:
            return x