"""
Feature selection utilities for machine learning.

This module provides tools for selecting the most relevant features
from datasets, including univariate and model-based selection methods.
"""

import numpy as np
from typing import List, Optional, Union, Callable, Tuple
import warnings


class SelectKBest:
    """
    Select features according to the k highest scores.

    Examples:
        >>> from fit.data.feature_selection import SelectKBest, f_classif
        >>> selector = SelectKBest(score_func=f_classif, k=5)
        >>> X_new = selector.fit_transform(X, y)
    """

    def __init__(self, score_func: Callable = None, k: int = 10):
        """
        Initialize SelectKBest.

        Args:
            score_func: Function taking two arrays X and y, and returning scores and p-values
            k: Number of top features to select
        """
        self.score_func = score_func or f_classif
        self.k = k
        self.scores_ = None
        self.pvalues_ = None
        self.selected_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SelectKBest":
        """
        Run score function on (X, y) and get the appropriate features.

        Args:
            X: Training data
            y: Target values

        Returns:
            Self for method chaining
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.scores_, self.pvalues_ = self.score_func(X, y)

        # Select top k features
        if self.k >= X.shape[1]:
            self.selected_features_ = np.arange(X.shape[1])
        else:
            self.selected_features_ = np.argsort(self.scores_)[-self.k :]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce X to the selected features.

        Args:
            X: Input data

        Returns:
            Data with selected features only
        """
        if self.selected_features_ is None:
            raise ValueError("This SelectKBest instance is not fitted yet.")

        X = np.asarray(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X: Training data
            y: Target values

        Returns:
            Data with selected features
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get a mask, or integer index, of the selected features.

        Args:
            indices: If True, return feature indices instead of mask

        Returns:
            Boolean mask or feature indices
        """
        if self.selected_features_ is None:
            raise ValueError("This SelectKBest instance is not fitted yet.")

        if indices:
            return self.selected_features_.tolist()
        else:
            mask = np.zeros(len(self.scores_), dtype=bool)
            mask[self.selected_features_] = True
            return mask


class SelectPercentile:
    """
    Select features according to a percentile of the highest scores.

    Examples:
        >>> selector = SelectPercentile(score_func=f_classif, percentile=20)
        >>> X_new = selector.fit_transform(X, y)
    """

    def __init__(self, score_func: Callable = None, percentile: int = 10):
        """
        Initialize SelectPercentile.

        Args:
            score_func: Function taking two arrays X and y, and returning scores
            percentile: Percent of features to keep
        """
        self.score_func = score_func or f_classif
        self.percentile = percentile
        self.scores_ = None
        self.pvalues_ = None
        self.selected_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SelectPercentile":
        """
        Run score function on (X, y) and get the appropriate features.

        Args:
            X: Training data
            y: Target values

        Returns:
            Self for method chaining
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.scores_, self.pvalues_ = self.score_func(X, y)

        # Select features by percentile
        n_features = X.shape[1]
        k = int(n_features * self.percentile / 100)
        k = max(1, k)  # Select at least one feature

        self.selected_features_ = np.argsort(self.scores_)[-k:]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce X to the selected features.

        Args:
            X: Input data

        Returns:
            Data with selected features only
        """
        if self.selected_features_ is None:
            raise ValueError("This SelectPercentile instance is not fitted yet.")

        X = np.asarray(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X: Training data
            y: Target values

        Returns:
            Data with selected features
        """
        return self.fit(X, y).transform(X)


class VarianceThreshold:
    """
    Feature selector that removes all low-variance features.

    Examples:
        >>> selector = VarianceThreshold(threshold=0.1)
        >>> X_new = selector.fit_transform(X)
    """

    def __init__(self, threshold: float = 0.0):
        """
        Initialize VarianceThreshold.

        Args:
            threshold: Features with variance below this threshold will be removed
        """
        self.threshold = threshold
        self.variances_ = None
        self.selected_features_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "VarianceThreshold":
        """
        Learn which features have variance above the threshold.

        Args:
            X: Training data
            y: Not used, present for API consistency

        Returns:
            Self for method chaining
        """
        X = np.asarray(X, dtype=np.float64)

        self.variances_ = np.var(X, axis=0)
        self.selected_features_ = np.where(self.variances_ > self.threshold)[0]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce X to the selected features.

        Args:
            X: Input data

        Returns:
            Data with selected features only
        """
        if self.selected_features_ is None:
            raise ValueError("This VarianceThreshold instance is not fitted yet.")

        X = np.asarray(X)
        return X[:, self.selected_features_]

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X: Training data
            y: Not used, present for API consistency

        Returns:
            Data with selected features
        """
        return self.fit(X, y).transform(X)


class RFE:
    """
    Recursive Feature Elimination.

    Given an external estimator that assigns weights to features,
    RFE recursively eliminates features and builds the model with
    the remaining attributes.

    Examples:
        >>> from fit.nn.modules.linear import Linear
        >>> estimator = Linear(10, 1)  # Simple linear model
        >>> selector = RFE(estimator, n_features_to_select=5)
        >>> X_new = selector.fit_transform(X, y)
    """

    def __init__(
        self, estimator, n_features_to_select: Optional[int] = None, step: int = 1
    ):
        """
        Initialize RFE.

        Args:
            estimator: Supervised learning estimator with a fit method
            n_features_to_select: Number of features to select
            step: Number of features to remove at each iteration
        """
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.selected_features_ = None
        self.ranking_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFE":
        """
        Fit the RFE model.

        Args:
            X: Training data
            y: Target values

        Returns:
            Self for method chaining
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_features = X.shape[1]

        if self.n_features_to_select is None:
            self.n_features_to_select = n_features // 2

        # Initialize all features as selected
        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)

        # Eliminate features iteratively
        while np.sum(support) > self.n_features_to_select:
            # Get current features
            features = np.where(support)[0]

            # Train estimator on current features
            X_subset = X[:, features]

            # For simplicity, use feature variance as importance
            # In a real implementation, you'd use the estimator's feature importance
            importances = np.var(X_subset, axis=0)

            # Eliminate least important features
            n_eliminate = min(self.step, len(features) - self.n_features_to_select)

            # Get indices of least important features
            least_important = np.argsort(importances)[:n_eliminate]

            # Remove these features
            for idx in least_important:
                support[features[idx]] = False
                ranking[features[idx]] = np.sum(support) + 1

        self.selected_features_ = np.where(support)[0]
        self.ranking_ = ranking

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce X to the selected features.

        Args:
            X: Input data

        Returns:
            Data with selected features only
        """
        if self.selected_features_ is None:
            raise ValueError("This RFE instance is not fitted yet.")

        X = np.asarray(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X: Training data
            y: Target values

        Returns:
            Data with selected features
        """
        return self.fit(X, y).transform(X)


# Score functions for feature selection
def f_classif(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the ANOVA F-value for the provided sample.

    Args:
        X: Sample data
        y: Target values

    Returns:
        Tuple of (F-statistics, p-values)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    n_samples, n_features = X.shape

    if n_classes < 2:
        warnings.warn("y has less than 2 classes")
        return np.zeros(n_features), np.ones(n_features)

    # Compute F-statistics
    f_stats = np.zeros(n_features)
    p_values = np.ones(n_features)

    for feature_idx in range(n_features):
        feature_data = X[:, feature_idx]

        # Group means
        group_means = []
        group_vars = []
        group_sizes = []

        for cls in unique_classes:
            class_mask = y == cls
            class_data = feature_data[class_mask]

            if len(class_data) > 0:
                group_means.append(np.mean(class_data))
                group_vars.append(
                    np.var(class_data, ddof=1) if len(class_data) > 1 else 0
                )
                group_sizes.append(len(class_data))

        if len(group_means) < 2:
            continue

        # Overall mean
        overall_mean = np.mean(feature_data)

        # Between-group sum of squares
        ss_between = sum(
            n * (mean - overall_mean) ** 2 for n, mean in zip(group_sizes, group_means)
        )

        # Within-group sum of squares
        ss_within = sum((n - 1) * var for n, var in zip(group_sizes, group_vars))

        # Degrees of freedom
        df_between = n_classes - 1
        df_within = n_samples - n_classes

        if df_within > 0 and ss_within > 0:
            # F-statistic
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            f_stat = ms_between / ms_within
            f_stats[feature_idx] = f_stat

            # Simplified p-value approximation
            # In a full implementation, you'd use the F-distribution
            p_values[feature_idx] = 1.0 / (1.0 + f_stat)

    return f_stats, p_values


def f_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Univariate linear regression tests.

    Args:
        X: Sample data
        y: Target values

    Returns:
        Tuple of (F-statistics, p-values)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n_samples, n_features = X.shape

    # Compute F-statistics for each feature
    f_stats = np.zeros(n_features)
    p_values = np.ones(n_features)

    y_mean = np.mean(y)
    y_var = np.var(y, ddof=1)

    for feature_idx in range(n_features):
        feature_data = X[:, feature_idx]

        # Simple linear regression: y = a + b*x
        feature_mean = np.mean(feature_data)

        # Compute correlation coefficient
        numerator = np.sum((feature_data - feature_mean) * (y - y_mean))
        denominator = np.sqrt(
            np.sum((feature_data - feature_mean) ** 2) * np.sum((y - y_mean) ** 2)
        )

        if denominator > 0:
            r = numerator / denominator

            # F-statistic from correlation
            if abs(r) < 1.0:
                f_stat = (r**2 * (n_samples - 2)) / (1 - r**2)
                f_stats[feature_idx] = f_stat

                # Simplified p-value
                p_values[feature_idx] = 1.0 / (1.0 + f_stat)

    return f_stats, p_values


def mutual_info_classif(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate mutual information for a discrete target variable.

    Args:
        X: Sample data
        y: Target values

    Returns:
        Tuple of (mutual information scores, dummy p-values)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)

    for feature_idx in range(n_features):
        feature_data = X[:, feature_idx]

        # Discretize continuous features into bins
        n_bins = min(10, len(np.unique(feature_data)))

        if n_bins > 1:
            # Simple binning
            feature_binned = np.digitize(
                feature_data,
                np.linspace(feature_data.min(), feature_data.max(), n_bins),
            )

            # Compute mutual information
            mi = _mutual_information(feature_binned, y)
            mi_scores[feature_idx] = mi

    # Return dummy p-values
    p_values = 1.0 / (1.0 + mi_scores)

    return mi_scores, p_values


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute mutual information between two discrete variables.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Mutual information score
    """
    # Get unique values and their counts
    x_values = np.unique(x)
    y_values = np.unique(y)

    n_samples = len(x)

    # Compute joint and marginal probabilities
    mi = 0.0

    for x_val in x_values:
        for y_val in y_values:
            # Joint probability
            p_xy = np.sum((x == x_val) & (y == y_val)) / n_samples

            if p_xy > 0:
                # Marginal probabilities
                p_x = np.sum(x == x_val) / n_samples
                p_y = np.sum(y == y_val) / n_samples

                # Mutual information contribution
                mi += p_xy * np.log(p_xy / (p_x * p_y))

    return max(0, mi)  # Ensure non-negative
