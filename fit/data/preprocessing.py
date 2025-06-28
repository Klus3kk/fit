"""
Data preprocessing utilities for machine learning.

This module provides tools for encoding categorical variables,
scaling features, and other preprocessing tasks.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
import warnings


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1.
    
    Examples:
        >>> encoder = LabelEncoder()
        >>> labels = ['cat', 'dog', 'cat', 'bird']
        >>> encoded = encoder.fit_transform(labels)
        >>> print(encoded)  # [0, 1, 0, 2]
        >>> decoded = encoder.inverse_transform(encoded)
        >>> print(decoded)  # ['cat', 'dog', 'cat', 'bird']
    """
    
    def __init__(self):
        self.classes_ = None
        self.class_to_index = None
        
    def fit(self, y: Union[List, np.ndarray]) -> 'LabelEncoder':
        """
        Fit label encoder.
        
        Args:
            y: Target values
            
        Returns:
            Self for method chaining
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self
        
    def transform(self, y: Union[List, np.ndarray]) -> np.ndarray:
        """
        Transform labels to normalized encoding.
        
        Args:
            y: Target values
            
        Returns:
            Encoded labels
        """
        if self.classes_ is None:
            raise ValueError("This LabelEncoder instance is not fitted yet.")
            
        y = np.asarray(y)
        encoded = np.zeros(len(y), dtype=int)
        
        for i, label in enumerate(y):
            if label not in self.class_to_index:
                raise ValueError(f"Label '{label}' not seen during fit.")
            encoded[i] = self.class_to_index[label]
            
        return encoded
        
    def fit_transform(self, y: Union[List, np.ndarray]) -> np.ndarray:
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
            y: Encoded target values
            
        Returns:
            Original labels
        """
        if self.classes_ is None:
            raise ValueError("This LabelEncoder instance is not fitted yet.")
            
        y = np.asarray(y)
        return self.classes_[y]


class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.
    
    Examples:
        >>> encoder = OneHotEncoder()
        >>> data = [['cat'], ['dog'], ['cat'], ['bird']]
        >>> encoded = encoder.fit_transform(data)
        >>> print(encoded.shape)  # (4, 3)
    """
    
    def __init__(self, sparse: bool = False, drop: Optional[str] = None):
        """
        Initialize OneHotEncoder.
        
        Args:
            sparse: Return sparse matrix if True (not implemented yet)
            drop: Strategy to use to drop one category per feature (not implemented yet)
        """
        self.sparse = sparse
        self.drop = drop
        self.categories_ = None
        self.feature_names_in_ = None
        
        if sparse:
            warnings.warn("Sparse output not implemented yet, will return dense array")
            
    def fit(self, X: Union[List, np.ndarray]) -> 'OneHotEncoder':
        """
        Fit OneHotEncoder to X.
        
        Args:
            X: Input samples
            
        Returns:
            Self for method chaining
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        self.categories_ = []
        
        for col_idx in range(X.shape[1]):
            unique_vals = np.unique(X[:, col_idx])
            self.categories_.append(unique_vals)
            
        return self
        
    def transform(self, X: Union[List, np.ndarray]) -> np.ndarray:
        """
        Transform X using one-hot encoding.
        
        Args:
            X: Input samples
            
        Returns:
            One-hot encoded array
        """
        if self.categories_ is None:
            raise ValueError("This OneHotEncoder instance is not fitted yet.")
            
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        encoded_features = []
        
        for col_idx in range(X.shape[1]):
            categories = self.categories_[col_idx]
            col_data = X[:, col_idx]
            
            # Create one-hot for this column
            col_encoded = np.zeros((len(col_data), len(categories)))
            
            for i, value in enumerate(col_data):
                if value in categories:
                    cat_idx = np.where(categories == value)[0][0]
                    col_encoded[i, cat_idx] = 1
                else:
                    raise ValueError(f"Value '{value}' not seen during fit.")
                    
            encoded_features.append(col_encoded)
            
        return np.hstack(encoded_features)
        
    def fit_transform(self, X: Union[List, np.ndarray]) -> np.ndarray:
        """
        Fit OneHotEncoder and transform X.
        
        Args:
            X: Input samples
            
        Returns:
            One-hot encoded array
        """
        return self.fit(X).transform(X)
        
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.
        
        Args:
            input_features: Input feature names
            
        Returns:
            Output feature names
        """
        if self.categories_ is None:
            raise ValueError("This OneHotEncoder instance is not fitted yet.")
            
        if input_features is None:
            input_features = [f"x{i}" for i in range(len(self.categories_))]
            
        feature_names = []
        for feature_idx, categories in enumerate(self.categories_):
            feature_name = input_features[feature_idx]
            for category in categories:
                feature_names.append(f"{feature_name}_{category}")
                
        return feature_names


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Examples:
        >>> scaler = StandardScaler()
        >>> X = [[1, 2], [3, 4], [5, 6]]
        >>> X_scaled = scaler.fit_transform(X)
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """
        Initialize StandardScaler.
        
        Args:
            with_mean: Center the data before scaling
            with_std: Scale the data to unit variance
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute the mean and std to be used for later scaling.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        X = np.asarray(X, dtype=np.float64)
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
            
        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            # Avoid division by zero
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.scale_ = np.ones(X.shape[1])
            
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("This StandardScaler instance is not fitted yet.")
            
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        Args:
            X: Training data
            
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
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("This StandardScaler instance is not fitted yet.")
            
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.mean_


class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range.
    
    Examples:
        >>> scaler = MinMaxScaler()
        >>> X = [[1, 2], [3, 4], [5, 6]]
        >>> X_scaled = scaler.fit_transform(X)  # Scale to [0, 1]
    """
    
    def __init__(self, feature_range: tuple = (0, 1)):
        """
        Initialize MinMaxScaler.
        
        Args:
            feature_range: Desired range of transformed data
        """
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute the minimum and maximum to be used for later scaling.
        
        Args:
            X: Training data
            
        Returns:
            Self for method chaining
        """
        X = np.asarray(X, dtype=np.float64)
        
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        
        data_range = self.data_max_ - self.data_min_
        # Avoid division by zero
        data_range[data_range == 0] = 1.0
        
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / data_range
        self.min_ = feature_min - self.data_min_ * self.scale_
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features according to feature_range.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if self.scale_ is None or self.min_ is None:
            raise ValueError("This MinMaxScaler instance is not fitted yet.")
            
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.min_
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        Args:
            X: Training data
            
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
        if self.scale_ is None or self.min_ is None:
            raise ValueError("This MinMaxScaler instance is not fitted yet.")
            
        X = np.asarray(X, dtype=np.float64)
        return (X - self.min_) / self.scale_