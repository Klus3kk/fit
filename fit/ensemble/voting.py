"""
Voting ensemble methods for model combination.

This module provides voting classifiers and regressors that combine
multiple models to improve prediction accuracy and robustness.
"""

import numpy as np
from typing import List, Union, Optional, Dict, Any, Tuple
from copy import deepcopy
import warnings


class VotingClassifier:
    """
    Soft and hard voting classifier for combining multiple models.

    In hard voting, the predicted class is the one that gets the most votes.
    In soft voting, the predicted class probabilities are averaged.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, Any]],
        voting: str = "hard",
        weights: Optional[List[float]] = None,
        n_jobs: Optional[int] = None,
        flatten_transform: bool = True,
    ):
        """
        Initialize voting classifier.

        Args:
            estimators: List of (name, estimator) tuples
            voting: Voting method ('hard' or 'soft')
            weights: Sequence of weights for each estimator
            n_jobs: Number of jobs for parallel processing (not implemented)
            flatten_transform: Whether to flatten transform output
        """
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform

        self.estimators_ = None
        self.classes_ = None
        self.named_estimators = {}

        # Validate inputs
        if voting not in ["hard", "soft"]:
            raise ValueError("voting must be 'hard' or 'soft'")

        if weights is not None and len(weights) != len(estimators):
            raise ValueError("Number of weights must equal number of estimators")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingClassifier":
        """
        Fit all estimators.

        Args:
            X: Training data
            y: Target values

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Get unique classes
        self.classes_ = np.unique(y)

        # Clone and fit estimators
        self.estimators_ = []
        for name, estimator in self.estimators:
            cloned_estimator = deepcopy(estimator)

            # Handle different model interfaces
            if hasattr(cloned_estimator, "fit"):
                # Standard sklearn-like interface
                cloned_estimator.fit(X, y)
            else:
                # Handle FIT framework models
                try:
                    from fit.core.tensor import Tensor
                    from fit.data.dataset import Dataset
                    from fit.data.dataloader import DataLoader
                    from fit.simple.trainer import train

                    # Convert to tensors and train
                    train_data = (X, y)
                    tracker = train(
                        cloned_estimator, train_data, epochs=50, verbose=False
                    )
                except ImportError:
                    # Fallback if FIT modules not available
                    raise ValueError(
                        "Estimator must have a 'fit' method or be a FIT framework model"
                    )

            self.estimators_.append((name, cloned_estimator))
            self.named_estimators[name] = cloned_estimator

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input data

        Returns:
            Predicted class labels
        """
        if self.voting == "hard":
            return self._predict_hard(X)
        else:
            return self._predict_soft(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (only for soft voting).

        Args:
            X: Input data

        Returns:
            Predicted class probabilities
        """
        if self.voting != "soft":
            raise ValueError("predict_proba is only available for soft voting")

        return self._predict_proba(X)

    def _predict_hard(self, X: np.ndarray) -> np.ndarray:
        """Hard voting prediction."""
        X = np.asarray(X)
        predictions = self._collect_predictions(X)

        # Apply weights if provided
        if self.weights is not None:
            weighted_predictions = np.zeros_like(predictions, dtype=float)
            for i, weight in enumerate(self.weights):
                weighted_predictions[:, i] = predictions[:, i] * weight
            predictions = weighted_predictions

        # Majority vote
        votes = np.zeros((len(X), len(self.classes_)))
        for i, pred_col in enumerate(predictions.T):
            for j, class_label in enumerate(self.classes_):
                votes[:, j] += pred_col == class_label

        return self.classes_[np.argmax(votes, axis=1)]

    def _predict_soft(self, X: np.ndarray) -> np.ndarray:
        """Soft voting prediction."""
        probas = self._predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for soft voting."""
        X = np.asarray(X)
        n_samples = len(X)
        n_classes = len(self.classes_)

        # Collect probabilities from all estimators
        all_probas = []
        weights = (
            self.weights if self.weights is not None else [1.0] * len(self.estimators_)
        )

        for (name, estimator), weight in zip(self.estimators_, weights):
            if hasattr(estimator, "predict_proba"):
                # Standard sklearn-like interface
                probas = estimator.predict_proba(X)
            else:
                # Handle FIT framework models
                try:
                    from fit.core.tensor import Tensor

                    X_tensor = Tensor(X)
                    outputs = estimator(X_tensor)

                    if outputs.data.ndim > 1 and outputs.data.shape[1] > 1:
                        # Multiclass - apply softmax
                        exp_outputs = np.exp(
                            outputs.data - np.max(outputs.data, axis=1, keepdims=True)
                        )
                        probas = exp_outputs / np.sum(
                            exp_outputs, axis=1, keepdims=True
                        )
                    else:
                        # Binary - apply sigmoid
                        sigmoid_outputs = 1 / (1 + np.exp(-outputs.data.flatten()))
                        probas = np.column_stack([1 - sigmoid_outputs, sigmoid_outputs])
                except ImportError:
                    raise ValueError("Cannot get probabilities from this estimator")

            # Ensure probabilities match expected classes
            if probas.shape[1] != n_classes:
                # Handle case where estimator doesn't predict all classes
                full_probas = np.zeros((n_samples, n_classes))
                estimator_classes = getattr(estimator, "classes_", self.classes_)

                for i, cls in enumerate(estimator_classes):
                    if cls in self.classes_:
                        cls_idx = np.where(self.classes_ == cls)[0][0]
                        full_probas[:, cls_idx] = probas[:, i]

                probas = full_probas

            all_probas.append(probas * weight)

        # Average probabilities
        avg_probas = np.mean(all_probas, axis=0)

        # Normalize to ensure they sum to 1
        return avg_probas / np.sum(avg_probas, axis=1, keepdims=True)

    def _collect_predictions(self, X: np.ndarray) -> np.ndarray:
        """Collect predictions from all estimators."""
        X = np.asarray(X)
        predictions = []

        for name, estimator in self.estimators_:
            if hasattr(estimator, "predict"):
                # Standard sklearn-like interface
                pred = estimator.predict(X)
            else:
                # Handle FIT framework models
                try:
                    from fit.core.tensor import Tensor

                    X_tensor = Tensor(X)
                    outputs = estimator(X_tensor)

                    if outputs.data.ndim > 1 and outputs.data.shape[1] > 1:
                        # Multiclass
                        pred = np.argmax(outputs.data, axis=1)
                        pred = self.classes_[pred]  # Map to actual class labels
                    else:
                        # Binary
                        pred = (outputs.data.flatten() > 0.5).astype(int)
                        pred = self.classes_[pred]  # Map to actual class labels
                except ImportError:
                    raise ValueError("Cannot get predictions from this estimator")

            predictions.append(pred)

        return np.column_stack(predictions)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy.

        Args:
            X: Test samples
            y: True labels

        Returns:
            Mean accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class VotingRegressor:
    """
    Voting regressor for combining multiple regression models.

    Predictions are averaged (optionally weighted) across all estimators.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, Any]],
        weights: Optional[List[float]] = None,
        n_jobs: Optional[int] = None,
    ):
        """
        Initialize voting regressor.

        Args:
            estimators: List of (name, estimator) tuples
            weights: Sequence of weights for each estimator
            n_jobs: Number of jobs for parallel processing (not implemented)
        """
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs

        self.estimators_ = None
        self.named_estimators = {}

        if weights is not None and len(weights) != len(estimators):
            raise ValueError("Number of weights must equal number of estimators")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingRegressor":
        """
        Fit all estimators.

        Args:
            X: Training data
            y: Target values

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Clone and fit estimators
        self.estimators_ = []
        for name, estimator in self.estimators:
            cloned_estimator = deepcopy(estimator)

            # Handle different model interfaces
            if hasattr(cloned_estimator, "fit"):
                # Standard sklearn-like interface
                cloned_estimator.fit(X, y)
            else:
                # Handle FIT framework models
                try:
                    from fit.core.tensor import Tensor
                    from fit.data.dataset import Dataset
                    from fit.data.dataloader import DataLoader
                    from fit.simple.trainer import train

                    # Convert to tensors and train
                    train_data = (X, y)
                    tracker = train(
                        cloned_estimator, train_data, epochs=50, verbose=False
                    )
                except ImportError:
                    raise ValueError(
                        "Estimator must have a 'fit' method or be a FIT framework model"
                    )

            self.estimators_.append((name, cloned_estimator))
            self.named_estimators[name] = cloned_estimator

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression targets.

        Args:
            X: Input data

        Returns:
            Predicted targets
        """
        X = np.asarray(X)
        predictions = self._collect_predictions(X)

        # Apply weights if provided
        if self.weights is not None:
            weights = np.array(self.weights).reshape(1, -1)
            weighted_predictions = predictions * weights
            return np.sum(weighted_predictions, axis=1) / np.sum(self.weights)
        else:
            return np.mean(predictions, axis=1)

    def _collect_predictions(self, X: np.ndarray) -> np.ndarray:
        """Collect predictions from all estimators."""
        X = np.asarray(X)
        predictions = []

        for name, estimator in self.estimators_:
            if hasattr(estimator, "predict"):
                # Standard sklearn-like interface
                pred = estimator.predict(X)
            else:
                # Handle FIT framework models
                try:
                    from fit.core.tensor import Tensor

                    X_tensor = Tensor(X)
                    outputs = estimator(X_tensor)
                    pred = outputs.data.flatten()
                except ImportError:
                    raise ValueError("Cannot get predictions from this estimator")

            predictions.append(pred)

        return np.column_stack(predictions)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2.

        Args:
            X: Test samples
            y: True targets

        Returns:
            R^2 score
        """
        predictions = self.predict(X)

        # Calculate R^2 score
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)


class StackingClassifier:
    """
    Stacking classifier that uses a meta-learner to combine base models.

    The base models are trained on the full dataset, then a meta-learner
    is trained on the cross-validated predictions of the base models.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, Any]],
        final_estimator: Any,
        cv: int = 5,
        stack_method: str = "auto",
        n_jobs: Optional[int] = None,
        passthrough: bool = False,
    ):
        """
        Initialize stacking classifier.

        Args:
            estimators: List of (name, base_estimator) tuples
            final_estimator: Meta-learner estimator
            cv: Number of cross-validation folds
            stack_method: Method for stacking ('auto', 'predict_proba', 'decision_function', 'predict')
            n_jobs: Number of jobs for parallel processing (not implemented)
            passthrough: Whether to include original features for meta-learner
        """
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.passthrough = passthrough

        self.estimators_ = None
        self.final_estimator_ = None
        self.classes_ = None
        self.named_estimators = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingClassifier":
        """
        Fit the stacking classifier.

        Args:
            X: Training data
            y: Target values

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_samples = len(X)

        # Fit base estimators and collect cross-validated predictions
        self.estimators_ = []
        cv_predictions = []

        # Simple k-fold cross-validation
        fold_size = n_samples // self.cv

        for name, estimator in self.estimators:
            # Store fitted estimator (trained on full data)
            full_estimator = deepcopy(estimator)
            self._fit_estimator(full_estimator, X, y)
            self.estimators_.append((name, full_estimator))
            self.named_estimators[name] = full_estimator

            # Collect cross-validated predictions
            cv_pred = np.zeros((n_samples, len(self.classes_)))

            for fold in range(self.cv):
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < self.cv - 1 else n_samples

                # Split data
                train_idx = np.concatenate(
                    [np.arange(0, start_idx), np.arange(end_idx, n_samples)]
                )
                val_idx = np.arange(start_idx, end_idx)

                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]

                # Fit estimator on fold training data
                fold_estimator = deepcopy(estimator)
                self._fit_estimator(fold_estimator, X_train_fold, y_train_fold)

                # Get predictions for validation fold
                if self.stack_method == "predict_proba" or self.stack_method == "auto":
                    val_pred = self._get_probabilities(fold_estimator, X_val_fold)
                else:
                    val_pred = self._get_predictions(fold_estimator, X_val_fold)
                    # Convert to one-hot for stacking
                    val_pred_onehot = np.zeros((len(val_pred), len(self.classes_)))
                    for i, pred in enumerate(val_pred):
                        class_idx = np.where(self.classes_ == pred)[0][0]
                        val_pred_onehot[i, class_idx] = 1
                    val_pred = val_pred_onehot

                cv_pred[val_idx] = val_pred

            cv_predictions.append(cv_pred)

        # Prepare meta-features
        meta_features = np.hstack(cv_predictions)

        if self.passthrough:
            meta_features = np.hstack([X, meta_features])

        # Fit final estimator
        self.final_estimator_ = deepcopy(self.final_estimator)
        self._fit_estimator(self.final_estimator_, meta_features, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input data

        Returns:
            Predicted class labels
        """
        X = np.asarray(X)

        # Get predictions from base estimators
        base_predictions = []
        for name, estimator in self.estimators_:
            if self.stack_method == "predict_proba" or self.stack_method == "auto":
                pred = self._get_probabilities(estimator, X)
            else:
                pred = self._get_predictions(estimator, X)
                # Convert to one-hot
                pred_onehot = np.zeros((len(pred), len(self.classes_)))
                for i, p in enumerate(pred):
                    class_idx = np.where(self.classes_ == p)[0][0]
                    pred_onehot[i, class_idx] = 1
                pred = pred_onehot

            base_predictions.append(pred)

        # Prepare meta-features
        meta_features = np.hstack(base_predictions)

        if self.passthrough:
            meta_features = np.hstack([X, meta_features])

        # Get final prediction
        return self._get_predictions(self.final_estimator_, meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input data

        Returns:
            Predicted class probabilities
        """
        X = np.asarray(X)

        # Get predictions from base estimators
        base_predictions = []
        for name, estimator in self.estimators_:
            if self.stack_method == "predict_proba" or self.stack_method == "auto":
                pred = self._get_probabilities(estimator, X)
            else:
                pred = self._get_predictions(estimator, X)
                # Convert to one-hot
                pred_onehot = np.zeros((len(pred), len(self.classes_)))
                for i, p in enumerate(pred):
                    class_idx = np.where(self.classes_ == p)[0][0]
                    pred_onehot[i, class_idx] = 1
                pred = pred_onehot

            base_predictions.append(pred)

        # Prepare meta-features
        meta_features = np.hstack(base_predictions)

        if self.passthrough:
            meta_features = np.hstack([X, meta_features])

        # Get final probabilities
        return self._get_probabilities(self.final_estimator_, meta_features)

    def _fit_estimator(self, estimator, X, y):
        """Fit an estimator with proper interface handling."""
        if hasattr(estimator, "fit"):
            estimator.fit(X, y)
        else:
            # Handle FIT framework models
            try:
                from fit.simple.trainer import train

                train_data = (X, y)
                train(estimator, train_data, epochs=50, verbose=False)
            except ImportError:
                raise ValueError(
                    "Estimator must have a 'fit' method or be a FIT framework model"
                )

    def _get_predictions(self, estimator, X):
        """Get predictions from an estimator."""
        if hasattr(estimator, "predict"):
            return estimator.predict(X)
        else:
            # Handle FIT framework models
            try:
                from fit.core.tensor import Tensor

                X_tensor = Tensor(X)
                outputs = estimator(X_tensor)

                if outputs.data.ndim > 1 and outputs.data.shape[1] > 1:
                    pred = np.argmax(outputs.data, axis=1)
                    return self.classes_[pred]
                else:
                    pred = (outputs.data.flatten() > 0.5).astype(int)
                    return self.classes_[pred]
            except ImportError:
                raise ValueError("Cannot get predictions from this estimator")

    def _get_probabilities(self, estimator, X):
        """Get probabilities from an estimator."""
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(X)
        else:
            # Handle FIT framework models
            try:
                from fit.core.tensor import Tensor

                X_tensor = Tensor(X)
                outputs = estimator(X_tensor)

                if outputs.data.ndim > 1 and outputs.data.shape[1] > 1:
                    # Multiclass - apply softmax
                    exp_outputs = np.exp(
                        outputs.data - np.max(outputs.data, axis=1, keepdims=True)
                    )
                    return exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
                else:
                    # Binary - apply sigmoid
                    sigmoid_outputs = 1 / (1 + np.exp(-outputs.data.flatten()))
                    return np.column_stack([1 - sigmoid_outputs, sigmoid_outputs])
            except ImportError:
                raise ValueError("Cannot get probabilities from this estimator")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy.

        Args:
            X: Test samples
            y: True labels

        Returns:
            Mean accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class StackingRegressor:
    """
    Stacking regressor that uses a meta-learner to combine base models.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, Any]],
        final_estimator: Any,
        cv: int = 5,
        n_jobs: Optional[int] = None,
        passthrough: bool = False,
    ):
        """
        Initialize stacking regressor.

        Args:
            estimators: List of (name, base_estimator) tuples
            final_estimator: Meta-learner estimator
            cv: Number of cross-validation folds
            n_jobs: Number of jobs for parallel processing (not implemented)
            passthrough: Whether to include original features for meta-learner
        """
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self.passthrough = passthrough

        self.estimators_ = None
        self.final_estimator_ = None
        self.named_estimators = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingRegressor":
        """
        Fit the stacking regressor.

        Args:
            X: Training data
            y: Target values

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = len(X)

        # Fit base estimators and collect cross-validated predictions
        self.estimators_ = []
        cv_predictions = []

        # Simple k-fold cross-validation
        fold_size = n_samples // self.cv

        for name, estimator in self.estimators:
            # Store fitted estimator (trained on full data)
            full_estimator = deepcopy(estimator)
            self._fit_estimator(full_estimator, X, y)
            self.estimators_.append((name, full_estimator))
            self.named_estimators[name] = full_estimator

            # Collect cross-validated predictions
            cv_pred = np.zeros(n_samples)

            for fold in range(self.cv):
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < self.cv - 1 else n_samples

                # Split data
                train_idx = np.concatenate(
                    [np.arange(0, start_idx), np.arange(end_idx, n_samples)]
                )
                val_idx = np.arange(start_idx, end_idx)

                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]

                # Fit estimator on fold training data
                fold_estimator = deepcopy(estimator)
                self._fit_estimator(fold_estimator, X_train_fold, y_train_fold)

                # Get predictions for validation fold
                val_pred = self._get_predictions(fold_estimator, X_val_fold)
                cv_pred[val_idx] = val_pred

            cv_predictions.append(cv_pred)

        # Prepare meta-features
        meta_features = np.column_stack(cv_predictions)

        if self.passthrough:
            meta_features = np.hstack([X, meta_features])

        # Fit final estimator
        self.final_estimator_ = deepcopy(self.final_estimator)
        self._fit_estimator(self.final_estimator_, meta_features, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression targets.

        Args:
            X: Input data

        Returns:
            Predicted targets
        """
        X = np.asarray(X)

        # Get predictions from base estimators
        base_predictions = []
        for name, estimator in self.estimators_:
            pred = self._get_predictions(estimator, X)
            base_predictions.append(pred)

        # Prepare meta-features
        meta_features = np.column_stack(base_predictions)

        if self.passthrough:
            meta_features = np.hstack([X, meta_features])

        # Get final prediction
        return self._get_predictions(self.final_estimator_, meta_features)

    def _fit_estimator(self, estimator, X, y):
        """Fit an estimator with proper interface handling."""
        if hasattr(estimator, "fit"):
            estimator.fit(X, y)
        else:
            # Handle FIT framework models
            try:
                from fit.simple.trainer import train

                train_data = (X, y)
                train(estimator, train_data, epochs=50, verbose=False)
            except ImportError:
                raise ValueError(
                    "Estimator must have a 'fit' method or be a FIT framework model"
                )

    def _get_predictions(self, estimator, X):
        """Get predictions from an estimator."""
        if hasattr(estimator, "predict"):
            return estimator.predict(X)
        else:
            # Handle FIT framework models
            try:
                from fit.core.tensor import Tensor

                X_tensor = Tensor(X)
                outputs = estimator(X_tensor)
                return outputs.data.flatten()
            except ImportError:
                raise ValueError("Cannot get predictions from this estimator")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2.

        Args:
            X: Test samples
            y: True targets

        Returns:
            R^2 score
        """
        predictions = self.predict(X)

        # Calculate R^2 score
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)
