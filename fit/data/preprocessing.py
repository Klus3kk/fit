"""
Data preprocessing utilities for machine learning workflows.

This module provides scalers, encoders, and data cleaning utilities
commonly needed for Kaggle competitions and ML projects.
"""

import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample x is calculated as:
    z = (x - u) / s
    where u is the mean and s is the standard deviation.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Compute the mean and std to be used for later scaling.

        Args:
            X: Training data to compute statistics on

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        # Handle constant features (std = 0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.

        Args:
            X: Data to transform

        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X: Data to fit and transform

        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale back the data to the original representation.

        Args:
            X: Transformed data

        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")

        X = np.asarray(X)
        return X * self.scale_ + self.mean_


class MinMaxScaler:
    """
    Scale features to a given range, usually [0, 1].

    The transformation is given by:
    X_scaled = (X - X.min) / (X.max - X.min) * (max - min) + min
    """

    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """
        Compute the minimum and maximum to be used for later scaling.

        Args:
            X: Training data

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)

        # Compute scale and min
        data_range = self.data_max_ - self.data_min_
        # Handle constant features
        data_range = np.where(data_range == 0, 1.0, data_range)

        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / data_range
        self.min_ = feature_min - self.data_min_ * self.scale_

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features according to feature_range.

        Args:
            X: Data to transform

        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.min_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X: Data to fit and transform

        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Undo the scaling of X according to feature_range.

        Args:
            X: Transformed data

        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")

        X = np.asarray(X)
        return (X - self.min_) / self.scale_


class RobustScaler:
    """
    Scale features using statistics that are robust to outliers.

    Uses median and interquartile range instead of mean and std.
    """

    def __init__(self):
        self.center_ = None
        self.scale_ = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> "RobustScaler":
        """
        Compute the median and IQR to be used for later scaling.

        Args:
            X: Training data

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)

        self.center_ = np.median(X, axis=0)

        # Compute IQR (75th percentile - 25th percentile)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self.scale_ = q75 - q25

        # Handle constant features
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features using median and IQR.

        Args:
            X: Data to transform

        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        return (X - self.center_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X: Data to fit and transform

        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1.
    """

    def __init__(self):
        self.classes_ = None
        self.fitted = False

    def fit(self, y: np.ndarray) -> "LabelEncoder":
        """
        Fit label encoder.

        Args:
            y: Target values

        Returns:
            self
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform labels to normalized encoding.

        Args:
            y: Target values

        Returns:
            Encoded labels
        """
        if not self.fitted:
            raise ValueError("LabelEncoder has not been fitted yet.")

        y = np.asarray(y)

        # Create mapping from class to index
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}

        # Transform
        encoded = np.array([class_to_idx.get(val, -1) for val in y])

        # Check for unknown classes
        if np.any(encoded == -1):
            unknown_classes = np.unique(y[encoded == -1])
            raise ValueError(f"Unknown classes found: {unknown_classes}")

        return encoded

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Fit label encoder and return encoded labels.

        Args:
            y: Target values

        Returns:
            Encoded labels
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform labels back to original encoding.

        Args:
            y: Encoded labels

        Returns:
            Original labels
        """
        if not self.fitted:
            raise ValueError("LabelEncoder has not been fitted yet.")

        y = np.asarray(y, dtype=int)

        if np.any((y < 0) | (y >= len(self.classes_))):
            raise ValueError("Invalid encoded labels found.")

        return self.classes_[y]


class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.
    """

    def __init__(self, drop_first: bool = False, sparse: bool = False):
        self.drop_first = drop_first
        self.sparse = sparse
        self.categories_ = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        """
        Fit OneHotEncoder to X.

        Args:
            X: Input data with categorical features

        Returns:
            self
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.categories_ = []
        for col in range(X.shape[1]):
            unique_vals = np.unique(X[:, col])
            self.categories_.append(unique_vals)

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X using one-hot encoding.

        Args:
            X: Input data

        Returns:
            One-hot encoded data
        """
        if not self.fitted:
            raise ValueError("OneHotEncoder has not been fitted yet.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        encoded_columns = []

        for col in range(X.shape[1]):
            categories = self.categories_[col]

            # Create one-hot encoding for this column
            col_data = X[:, col]
            one_hot = np.zeros((len(col_data), len(categories)))

            for i, val in enumerate(col_data):
                if val in categories:
                    idx = np.where(categories == val)[0][0]
                    one_hot[i, idx] = 1
                else:
                    warnings.warn(f"Unknown category '{val}' found during transform")

            # Drop first column if specified
            if self.drop_first and one_hot.shape[1] > 1:
                one_hot = one_hot[:, 1:]

            encoded_columns.append(one_hot)

        return np.hstack(encoded_columns)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit OneHotEncoder to X, then transform X.

        Args:
            X: Input data

        Returns:
            One-hot encoded data
        """
        return self.fit(X).transform(X)


def handle_missing_values(
    X: np.ndarray,
    strategy: str = "mean",
    fill_value: Optional[Union[str, float]] = None,
) -> np.ndarray:
    """
    Handle missing values in the dataset.

    Args:
        X: Input data with possible missing values (NaN)
        strategy: Strategy for handling missing values
                 ('mean', 'median', 'mode', 'constant', 'drop')
        fill_value: Value to use when strategy is 'constant'

    Returns:
        Data with missing values handled
    """
    X = np.asarray(X, dtype=float)

    if strategy == "drop":
        # Remove rows with any missing values
        return X[~np.isnan(X).any(axis=1)]

    elif strategy == "mean":
        # Fill with column means
        col_means = np.nanmean(X, axis=0)
        X_filled = X.copy()
        for col in range(X.shape[1]):
            mask = np.isnan(X_filled[:, col])
            X_filled[mask, col] = col_means[col]
        return X_filled

    elif strategy == "median":
        # Fill with column medians
        col_medians = np.nanmedian(X, axis=0)
        X_filled = X.copy()
        for col in range(X.shape[1]):
            mask = np.isnan(X_filled[:, col])
            X_filled[mask, col] = col_medians[col]
        return X_filled

    elif strategy == "mode":
        # Fill with column modes (most frequent values)
        X_filled = X.copy()
        for col in range(X.shape[1]):
            mask = np.isnan(X_filled[:, col])
            if np.any(mask):
                # Find mode of non-missing values
                col_data = X_filled[~mask, col]
                if len(col_data) > 0:
                    unique_vals, counts = np.unique(col_data, return_counts=True)
                    mode_val = unique_vals[np.argmax(counts)]
                    X_filled[mask, col] = mode_val
                else:
                    X_filled[mask, col] = 0  # Fallback
        return X_filled

    elif strategy == "constant":
        # Fill with constant value
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy='constant'")
        X_filled = X.copy()
        X_filled[np.isnan(X_filled)] = fill_value
        return X_filled

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def detect_outliers(
    X: np.ndarray, method: str = "iqr", threshold: float = 1.5
) -> np.ndarray:
    """
    Detect outliers in the dataset.

    Args:
        X: Input data
        method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold for outlier detection

    Returns:
        Boolean mask where True indicates outliers
    """
    X = np.asarray(X)

    if method == "iqr":
        # Interquartile Range method
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outliers = (X < lower_bound) | (X > upper_bound)
        return np.any(outliers, axis=1)

    elif method == "zscore":
        # Z-score method
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        outliers = z_scores > threshold
        return np.any(outliers, axis=1)

    elif method == "modified_zscore":
        # Modified Z-score using median
        median = np.median(X, axis=0)
        mad = np.median(np.abs(X - median), axis=0)
        modified_z_scores = 0.6745 * (X - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
        return np.any(outliers, axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")


def encode_categorical_features(
    X: np.ndarray,
    categorical_columns: List[int],
    method: str = "onehot",
    drop_first: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Encode categorical features in the dataset.

    Args:
        X: Input data
        categorical_columns: Indices of categorical columns
        method: Encoding method ('onehot', 'label', 'target')
        drop_first: Whether to drop the first category in one-hot encoding

    Returns:
        Tuple of (encoded_data, encoders_dict)
    """
    X = np.asarray(X)
    encoded_columns = []
    encoders = {}

    # Process non-categorical columns first
    non_categorical = [i for i in range(X.shape[1]) if i not in categorical_columns]
    if non_categorical:
        encoded_columns.append(X[:, non_categorical])

    # Process categorical columns
    for col_idx in categorical_columns:
        col_data = X[:, col_idx]

        if method == "onehot":
            encoder = OneHotEncoder(drop_first=drop_first)
            encoded_col = encoder.fit_transform(col_data.reshape(-1, 1))
            encoded_columns.append(encoded_col)
            encoders[f"col_{col_idx}"] = encoder

        elif method == "label":
            encoder = LabelEncoder()
            encoded_col = encoder.fit_transform(col_data).reshape(-1, 1)
            encoded_columns.append(encoded_col.astype(float))
            encoders[f"col_{col_idx}"] = encoder

        else:
            raise ValueError(f"Unknown encoding method: {method}")

    # Combine all columns
    if encoded_columns:
        result = np.hstack(encoded_columns)
    else:
        result = np.empty((X.shape[0], 0))

    return result, encoders


def create_polynomial_features(
    X: np.ndarray,
    degree: int = 2,
    include_bias: bool = True,
    interaction_only: bool = False,
) -> np.ndarray:
    """
    Create polynomial features from input features.

    Args:
        X: Input features
        degree: Maximum degree of polynomial features
        include_bias: Whether to include bias (constant) term
        interaction_only: Whether to only include interaction features

    Returns:
        Polynomial features
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape

    if degree < 1:
        raise ValueError("degree must be >= 1")

    features = []

    # Add bias term
    if include_bias:
        features.append(np.ones((n_samples, 1)))

    # Add original features (degree 1)
    features.append(X)

    # Add higher degree features
    if degree > 1:
        for d in range(2, degree + 1):
            if interaction_only:
                # Only interaction terms (no powers of single features)
                from itertools import combinations_with_replacement

                for indices in combinations_with_replacement(range(n_features), d):
                    if len(set(indices)) > 1:  # Only interactions
                        feature = np.ones((n_samples, 1))
                        for idx in indices:
                            feature = feature * X[:, idx : idx + 1]
                        features.append(feature)
            else:
                # All polynomial terms including powers
                from itertools import combinations_with_replacement

                for indices in combinations_with_replacement(range(n_features), d):
                    feature = np.ones((n_samples, 1))
                    for idx in indices:
                        feature = feature * X[:, idx : idx + 1]
                    features.append(feature)

    return np.hstack(features)


# Convenience function for complete preprocessing pipeline
def preprocess_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    categorical_columns: Optional[List[int]] = None,
    scaling_method: str = "standard",
    missing_strategy: str = "mean",
    encode_labels: bool = True,
    remove_outliers: bool = False,
    outlier_method: str = "iqr",
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Complete preprocessing pipeline for machine learning data.

    Args:
        X: Input features
        y: Target values (optional)
        categorical_columns: Indices of categorical columns
        scaling_method: Method for scaling ('standard', 'minmax', 'robust', 'none')
        missing_strategy: Strategy for missing values
        encode_labels: Whether to encode labels (if y is provided)
        remove_outliers: Whether to remove outliers
        outlier_method: Method for outlier detection

    Returns:
        Tuple of (processed_X, processed_y, preprocessors_dict)
    """
    X = np.asarray(X)
    preprocessors = {}

    # Handle missing values
    if np.any(np.isnan(X)):
        X = handle_missing_values(X, strategy=missing_strategy)

    # Remove outliers
    if remove_outliers:
        outlier_mask = detect_outliers(X, method=outlier_method)
        X = X[~outlier_mask]
        if y is not None:
            y = y[~outlier_mask]
        preprocessors["outliers_removed"] = np.sum(outlier_mask)

    # Encode categorical features
    if categorical_columns:
        X, encoders = encode_categorical_features(X, categorical_columns)
        preprocessors["categorical_encoders"] = encoders

    # Scale features
    if scaling_method != "none":
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")

        X = scaler.fit_transform(X)
        preprocessors["scaler"] = scaler

    # Encode labels
    processed_y = y
    if y is not None and encode_labels:
        # Check if y contains non-numeric values
        try:
            y_numeric = np.asarray(y, dtype=float)
            if not np.array_equal(y_numeric, y_numeric.astype(int)):
                # Continuous values, no encoding needed
                processed_y = y_numeric
            else:
                # Discrete values, might need encoding
                unique_vals = np.unique(y)
                if len(unique_vals) > 2 and not all(
                    isinstance(v, (int, np.integer)) for v in unique_vals
                ):
                    label_encoder = LabelEncoder()
                    processed_y = label_encoder.fit_transform(y)
                    preprocessors["label_encoder"] = label_encoder
                else:
                    processed_y = y_numeric.astype(int)
        except (ValueError, TypeError):
            # Non-numeric labels, definitely need encoding
            label_encoder = LabelEncoder()
            processed_y = label_encoder.fit_transform(y)
            preprocessors["label_encoder"] = label_encoder

    return X, processed_y, preprocessors
