"""
Cross-validation utilities for model selection and evaluation.

This module provides various cross-validation strategies commonly used
in machine learning and Kaggle competitions.
"""

import numpy as np
from typing import Generator, Tuple, List, Optional, Union, Dict, Any, Callable
import warnings


class KFoldCV:
    """
    K-Fold cross-validation iterator.
    
    Provides train/test indices to split data in train/test sets.
    Split dataset into k consecutive folds (without shuffling by default).
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        """
        Initialize K-Fold cross-validation.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random state for reproducibility
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Training data
            y: Target values (not used in KFold but kept for compatibility)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.
        
        Returns:
            Number of splits
        """
        return self.n_splits


class StratifiedKFoldCV:
    """
    Stratified K-Fold cross-validation iterator.
    
    Provides train/test indices to split data in train/test sets while
    preserving the percentage of samples for each class.
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        """
        Initialize Stratified K-Fold cross-validation.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random state for reproducibility
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: np.ndarray, y: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Training data
            y: Target values
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if y is None:
            raise ValueError("y cannot be None for StratifiedKFold")
        
        y = np.asarray(y)
        n_samples = len(X)
        
        # Get unique classes and their counts
        unique_classes, y_indices, class_counts = np.unique(y, return_inverse=True, return_counts=True)
        n_classes = len(unique_classes)
        
        # Check if we can do stratification
        if np.min(class_counts) < self.n_splits:
            warnings.warn(
                f"The least populated class has only {np.min(class_counts)} members, "
                f"which is less than n_splits={self.n_splits}."
            )
        
        # Create indices for each class
        class_indices = [np.where(y_indices == i)[0] for i in range(n_classes)]
        
        # Shuffle class indices if requested
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            for indices in class_indices:
                np.random.shuffle(indices)
        
        # Create fold assignments for each class
        fold_assignments = []
        for class_idx in class_indices:
            class_size = len(class_idx)
            fold_sizes = np.full(self.n_splits, class_size // self.n_splits, dtype=int)
            fold_sizes[:class_size % self.n_splits] += 1
            
            assignments = []
            for fold_id, fold_size in enumerate(fold_sizes):
                assignments.extend([fold_id] * fold_size)
            
            fold_assignments.append(np.array(assignments))
        
        # Generate splits
        for fold_id in range(self.n_splits):
            test_indices = []
            train_indices = []
            
            for class_id, (class_idx, assignments) in enumerate(zip(class_indices, fold_assignments)):
                mask = assignments == fold_id
                test_indices.extend(class_idx[mask])
                train_indices.extend(class_idx[~mask])
            
            yield np.array(train_indices), np.array(test_indices)
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.
        
        Returns:
            Number of splits
        """
        return self.n_splits


class TimeSeriesSplit:
    """
    Time Series cross-validation iterator.
    
    Provides train/test indices to split time series data sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross-validation is not meaningful.
    """
    
    def __init__(self, n_splits: int = 5, max_train_size: Optional[int] = None):
        """
        Initialize Time Series Split.
        
        Args:
            n_splits: Number of splits
            max_train_size: Maximum size for training set
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        
        self.n_splits = n_splits
        self.max_train_size = max_train_size
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Training data
            y: Target values (not used but kept for compatibility)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        
        if n_folds > n_samples:
            raise ValueError(f"Cannot have number of folds={n_folds} greater than number of samples={n_samples}")
        
        indices = np.arange(n_samples)
        test_size = n_samples // n_folds
        test_starts = range(test_size + n_samples % n_folds, n_samples, test_size)
        
        for test_start in test_starts:
            train_end = test_start
            test_end = test_start + test_size
            
            if self.max_train_size and self.max_train_size < train_end:
                train_start = train_end - self.max_train_size
            else:
                train_start = 0
            
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.
        
        Returns:
            Number of splits
        """
        return self.n_splits


class GroupKFoldCV:
    """
    K-fold iterator variant with non-overlapping groups.
    
    Ensures that the same group is not represented in both testing and training sets.
    """
    
    def __init__(self, n_splits: int = 5):
        """
        Initialize Group K-Fold cross-validation.
        
        Args:
            n_splits: Number of splits
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        
        self.n_splits = n_splits
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Training data
            y: Target values (not used but kept for compatibility)
            groups: Group labels for the samples
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")
        
        groups = np.asarray(groups)
        n_samples = len(X)
        
        if len(groups) != n_samples:
            raise ValueError("groups must have same length as X")
        
        # Get unique groups
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if n_groups < self.n_splits:
            raise ValueError(f"Number of groups ({n_groups}) is less than n_splits ({self.n_splits})")
        
        # Split groups into folds
        group_fold_counts = np.full(self.n_splits, n_groups // self.n_splits, dtype=int)
        group_fold_counts[:n_groups % self.n_splits] += 1
        
        # Assign groups to folds
        group_to_fold = {}
        current_fold = 0
        groups_in_current_fold = 0
        
        for group in unique_groups:
            if groups_in_current_fold >= group_fold_counts[current_fold]:
                current_fold += 1
                groups_in_current_fold = 0
            
            group_to_fold[group] = current_fold
            groups_in_current_fold += 1
        
        # Generate splits
        for fold_id in range(self.n_splits):
            test_groups = [group for group, fold in group_to_fold.items() if fold == fold_id]
            test_indices = np.where(np.isin(groups, test_groups))[0]
            train_indices = np.where(~np.isin(groups, test_groups))[0]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.
        
        Returns:
            Number of splits
        """
        return self.n_splits


class LeaveOneGroupOut:
    """
    Leave One Group Out cross-validation iterator.
    
    Provides train/test indices to split data according to a third-party
    provided group. This group information can be used to encode arbitrary
    domain specific pre-defined cross-validation folds.
    """
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Training data
            y: Target values (not used but kept for compatibility)
            groups: Group labels for the samples
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")
        
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            test_indices = np.where(groups == group)[0]
            train_indices = np.where(groups != group)[0]
            yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.
        
        Returns:
            Number of splits (number of unique groups)
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")
        
        groups = np.asarray(groups)
        return len(np.unique(groups))


def cross_validate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, Any] = 5,
    scoring: Union[str, Callable] = 'accuracy',
    groups: Optional[np.ndarray] = None,
    return_train_score: bool = False,
    return_estimator: bool = False,
    verbose: bool = False
) -> Dict[str, List[float]]:
    """
    Evaluate metric(s) by cross-validation and return scores for each fold.
    
    Args:
        model: Estimator object implementing 'fit' and 'predict'
        X: Training data
        y: Target values
        cv: Cross-validation splitting strategy (int or CV object)
        scoring: Scoring metric ('accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'mse', 'mae', 'r2')
        groups: Group labels for the samples (for GroupKFold)
        return_train_score: Whether to return training scores
        return_estimator: Whether to return fitted estimators
        verbose: Whether to print progress
        
    Returns:
        Dictionary with test scores and optionally train scores and estimators
    """
    from copy import deepcopy
    
    # Set up cross-validator
    if isinstance(cv, int):
        if scoring in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
            cv_splitter = StratifiedKFoldCV(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_splitter = KFoldCV(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_splitter = cv
    
    # Set up scoring function
    score_func = _get_scoring_function(scoring)
    
    # Storage for results
    results = {'test_score': []}
    if return_train_score:
        results['train_score'] = []
    if return_estimator:
        results['estimator'] = []
    
    # Perform cross-validation
    if hasattr(cv_splitter, 'split'):
        if groups is not None and hasattr(cv_splitter, 'split') and 'groups' in cv_splitter.split.__code__.co_varnames:
            splits = cv_splitter.split(X, y, groups=groups)
        else:
            splits = cv_splitter.split(X, y)
    else:
        raise ValueError("CV object must have a 'split' method")
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if verbose:
            print(f"Processing fold {fold_idx + 1}...")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone and fit model
        model_clone = deepcopy(model)
        
        # Handle different model interfaces
        if hasattr(model_clone, 'fit'):
            # Standard sklearn-like interface
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            
            if return_train_score:
                y_train_pred = model_clone.predict(X_train)
        else:
            # Handle FIT framework models
            from core.tensor import Tensor
            from data.dataset import Dataset
            from data.dataloader import DataLoader
            from simple.trainer import train
            
            # Convert to tensors and train
            train_dataset = Dataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            tracker = train(model_clone, train_loader, epochs=50, verbose=False)
            
            # Make predictions
            test_tensor = Tensor(X_test)
            y_pred_tensor = model_clone(test_tensor)
            
            if y_pred_tensor.data.ndim > 1 and y_pred_tensor.data.shape[1] > 1:
                y_pred = np.argmax(y_pred_tensor.data, axis=1)
            else:
                y_pred = y_pred_tensor.data.flatten()
            
            if return_train_score:
                train_tensor = Tensor(X_train)
                y_train_pred_tensor = model_clone(train_tensor)
                if y_train_pred_tensor.data.ndim > 1 and y_train_pred_tensor.data.shape[1] > 1:
                    y_train_pred = np.argmax(y_train_pred_tensor.data, axis=1)
                else:
                    y_train_pred = y_train_pred_tensor.data.flatten()
        
        # Calculate scores
        test_score = score_func(y_test, y_pred)
        results['test_score'].append(test_score)
        
        if return_train_score:
            train_score = score_func(y_train, y_train_pred)
            results['train_score'].append(train_score)
        
        if return_estimator:
            results['estimator'].append(model_clone)
        
        if verbose:
            print(f"Fold {fold_idx + 1} test score: {test_score:.4f}")
    
    if verbose:
        mean_score = np.mean(results['test_score'])
        std_score = np.std(results['test_score'])
        print(f"Cross-validation results: {mean_score:.4f} (+/- {std_score:.4f})")
    
    return results


def _get_scoring_function(scoring: Union[str, Callable]) -> Callable:
    """
    Get scoring function by name or return the function if already callable.
    
    Args:
        scoring: Scoring metric name or function
        
    Returns:
        Scoring function
    """
    if callable(scoring):
        return scoring
    
    if scoring == 'accuracy':
        return lambda y_true, y_pred: np.mean(y_true == y_pred)
    elif scoring == 'mse':
        return lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
    elif scoring == 'mae':
        return lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    elif scoring == 'r2':
        def r2_score(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        return r2_score
    elif scoring == 'f1':
        def f1_score(y_true, y_pred):
            # Binary F1 score
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score
    elif scoring == 'precision':
        def precision_score(y_true, y_pred):
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tp / (tp + fp) if (tp + fp) > 0 else 0
        return precision_score
    elif scoring == 'recall':
        def recall_score(y_true, y_pred):
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return tp / (tp + fn) if (tp + fn) > 0 else 0
        return recall_score
    else:
        raise ValueError(f"Unknown scoring metric: {scoring}")


def train_test_split(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    test_size: Union[float, int] = 0.2,
    train_size: Optional[Union[float, int]] = None,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Split arrays into random train and test subsets.
    
    Args:
        X: Input data
        y: Target values (optional)
        test_size: Proportion or absolute number of test samples
        train_size: Proportion or absolute number of train samples
        random_state: Random state for reproducibility
        shuffle: Whether to shuffle data before splitting
        stratify: Data to use for stratified splitting
        
    Returns:
        List of arrays: [X_train, X_test] or [X_train, X_test, y_train, y_test]
    """
    n_samples = len(X)
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calculate split sizes
    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    else:
        n_test = test_size
    
    if train_size is not None:
        if isinstance(train_size, float):
            n_train = int(n_samples * train_size)
        else:
            n_train = train_size
    else:
        n_train = n_samples - n_test
    
    if n_train + n_test > n_samples:
        raise ValueError("train_size + test_size cannot be greater than n_samples")
    
    # Generate indices
    indices = np.arange(n_samples)
    
    if stratify is not None:
        # Stratified split
        unique_classes = np.unique(stratify)
        train_indices = []
        test_indices = []
        
        for cls in unique_classes:
            cls_indices = indices[stratify == cls]
            cls_n_samples = len(cls_indices)
            cls_n_test = int(cls_n_samples * (n_test / n_samples))
            
            if shuffle:
                np.random.shuffle(cls_indices)
            
            test_indices.extend(cls_indices[:cls_n_test])
            train_indices.extend(cls_indices[cls_n_test:cls_n_test + int(cls_n_samples * (n_train / n_samples))])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
    else:
        # Regular split
        if shuffle:
            np.random.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:n_test + n_train]
    
    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    if y is not None:
        y_train = y[train_indices]
        y_test = y[test_indices]
        return [X_train, X_test, y_train, y_test]
    else:
        return [X_train, X_test]


# Convenience functions for specific use cases
def kaggle_time_series_split(
    df: np.ndarray,
    time_column_idx: int,
    test_size: float = 0.2,
    gap_size: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split time series data for Kaggle competitions.
    
    Args:
        df: Data array
        time_column_idx: Index of time column
        test_size: Proportion of data for testing
        gap_size: Number of samples to skip between train and test
        
    Returns:
        Tuple of (train_indices, test_indices)
    """
    n_samples = len(df)
    n_test = int(n_samples * test_size)
    
    # Sort by time
    time_column = df[:, time_column_idx]
    sorted_indices = np.argsort(time_column)
    
    # Create split with gap
    train_end = n_samples - n_test - gap_size
    train_indices = sorted_indices[:train_end]
    test_indices = sorted_indices[train_end + gap_size:]
    
    return train_indices, test_indices


def kaggle_group_split(
    X: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data by groups for Kaggle competitions.
    
    Args:
        X: Input data
        groups: Group labels
        test_size: Proportion of groups for testing
        
    Returns:
        Tuple of (train_indices, test_indices)
    """
    unique_groups = np.unique(groups)
    n_test_groups = int(len(unique_groups) * test_size)
    
    # Randomly select test groups
    np.random.shuffle(unique_groups)
    test_groups = unique_groups[:n_test_groups]
    
    test_indices = np.where(np.isin(groups, test_groups))[0]
    train_indices = np.where(~np.isin(groups, test_groups))[0]
    
    return train_indices, test_indices