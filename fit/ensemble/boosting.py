"""
Boosting ensemble methods.

This module implements boosting algorithms like AdaBoost that
sequentially fit weak learners and combine them into a strong learner.
"""

import numpy as np
from typing import Optional, Union, Any

from fit.core.tensor import Tensor
from fit.ensemble.base import BaseEnsemble


class AdaBoostClassifier(BaseEnsemble):
    """
    AdaBoost classifier implementation.
    
    AdaBoost fits a sequence of weak learners on repeatedly modified
    versions of the data. The predictions from all of them are then
    combined through a weighted majority vote.
    
    Examples:
        >>> from fit.ensemble import AdaBoostClassifier
        >>> from fit.simple.models import MLP
        >>> 
        >>> # Create AdaBoost classifier
        >>> ada = AdaBoostClassifier(
        ...     base_estimator=MLP([4, 2]),
        ...     n_estimators=50,
        ...     learning_rate=1.0
        ... )
        >>> ada.fit(X_train, y_train)
        >>> predictions = ada.predict(X_test)
    """
    
    def __init__(self, base_estimator=None, n_estimators: int = 50, 
                 learning_rate: float = 1.0, random_state: Optional[int] = None):
        """
        Initialize AdaBoost classifier.
        
        Args:
            base_estimator: Base estimator to boost
            n_estimators: Maximum number of estimators
            learning_rate: Learning rate shrinks the contribution of each classifier
            random_state: Random state for reproducibility
        """
        super().__init__(n_estimators=n_estimators, random_state=random_state)
        self.base_estimator = base_estimator
        self.learning_rate = learning_rate
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        
    def _make_estimator(self) -> Any:
        """
        Create a new estimator instance.
        
        Returns:
            New estimator instance
        """
        if self.base_estimator is None:
            # Default to a simple decision stump (single layer perceptron)
            from fit.simple.models import MLP
            return MLP([1, 1], activation="tanh")  # Simple weak learner
        
        # Create a copy of the base estimator
        if hasattr(self.base_estimator, 'copy'):
            return self.base_estimator.copy()
        else:
            estimator_class = self.base_estimator.__class__
            return estimator_class()
    
    def _fit_estimator(self, estimator, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray):
        """
        Fit a single estimator with sample weights.
        
        Args:
            estimator: The estimator to fit
            X: Training data
            y: Target values
            sample_weights: Weights for each sample
        """
        # For simplicity, we'll simulate weighted training by sampling
        # In a full implementation, the estimator would support sample weights
        
        # Create weighted bootstrap sample
        n_samples = len(X)
        weighted_indices = np.random.choice(
            n_samples, 
            size=n_samples, 
            replace=True, 
            p=sample_weights / sample_weights.sum()
        )
        
        X_weighted = X[weighted_indices]
        y_weighted = y[weighted_indices]
        
        # Fit estimator
        if hasattr(estimator, 'fit'):
            estimator.fit(X_weighted, y_weighted)
    
    def _predict_estimator(self, estimator, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a single estimator.
        
        Args:
            estimator: The fitted estimator
            X: Input data
            
        Returns:
            Predictions from the estimator
        """
        if hasattr(estimator, 'predict'):
            predictions = estimator.predict(X)
        elif hasattr(estimator, 'forward'):
            # For neural network models
            X_tensor = Tensor(X)
            predictions = estimator.forward(X_tensor).data
        else:
            raise ValueError(f"Estimator {estimator} has no predict or forward method")
        
        # Convert to binary predictions if needed
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
        
        # Convert to {-1, +1} format for AdaBoost
        unique_classes = np.unique(predictions)
        if len(unique_classes) == 2:
            # Binary classification: convert to -1, +1
            binary_pred = np.where(predictions == unique_classes[0], -1, 1)
            return binary_pred
        else:
            # Multi-class: keep original format
            return predictions
    
    def fit(self, X: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor]) -> 'AdaBoostClassifier':
        """
        Build a boosted classifier from the training set.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            Self for method chaining
        """
        # Convert to numpy if needed
        if isinstance(X, Tensor):
            X = X.data
        if isinstance(y, Tensor):
            y = y.data
            
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Convert labels to {-1, +1} for binary classification
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            y_binary = np.where(y == self.classes_[0], -1, 1)
        else:
            # For multi-class, we'll use one-vs-rest approach (simplified)
            y_binary = y.copy()
        
        n_samples = X.shape[0]
        
        # Initialize sample weights uniformly
        sample_weights = np.ones(n_samples) / n_samples
        
        # Clear previous results
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        
        for iboost in range(self.n_estimators):
            # Create and fit weak learner
            estimator = self._make_estimator()
            self._fit_estimator(estimator, X, y_binary, sample_weights)
            
            # Get predictions
            y_predict = self._predict_estimator(estimator, X)
            
            # Calculate error rate
            incorrect = y_predict != y_binary
            estimator_error = np.average(incorrect, weights=sample_weights)
            
            # If error is too high or too low, stop
            if estimator_error <= 0:
                # Perfect classifier
                self.estimators_.append(estimator)
                self.estimator_weights_.append(1.0)
                self.estimator_errors_.append(estimator_error)
                break
            
            if estimator_error >= 0.5:
                # Worse than random
                if len(self.estimators_) == 0:
                    raise ValueError("BaseClassifier in AdaBoostClassifier "
                                   "ensemble is worse than random, ensemble "
                                   "can not be fitted.")
                break
            
            # Calculate alpha (estimator weight)
            alpha = self.learning_rate * 0.5 * np.log((1 - estimator_error) / estimator_error)
            
            # Store estimator and its weight
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(estimator_error)
            
            # Update sample weights
            sample_weights *= np.exp(alpha * incorrect * (y_predict != y_binary))
            sample_weights /= sample_weights.sum()
            
            # If all samples have equal weight, stop
            if np.abs(sample_weights - 1.0 / n_samples).sum() < 1e-10:
                break
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Predict classes for samples in X.
        
        Args:
            X: Input data
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted_:
            raise ValueError("This AdaBoostClassifier instance is not fitted yet.")
        
        # Convert to numpy if needed
        if isinstance(X, Tensor):
            X = X.data
        X = np.asarray(X)
        
        n_samples = X.shape[0]
        
        # Get weighted predictions from all estimators
        decision_scores = np.zeros(n_samples)
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            predictions = self._predict_estimator(estimator, X)
            decision_scores += weight * predictions
        
        # Convert back to original class labels
        if len(self.classes_) == 2:
            # Binary classification
            binary_predictions = np.where(decision_scores >= 0, 1, -1)
            return np.where(binary_predictions == -1, self.classes_[0], self.classes_[1])
        else:
            # Multi-class (simplified)
            return np.where(decision_scores >= 0, self.classes_[1], self.classes_[0])
    
    def decision_function(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Compute the decision function of X.
        
        Args:
            X: Input data
            
        Returns:
            Decision function values
        """
        if not self.is_fitted_:
            raise ValueError("This AdaBoostClassifier instance is not fitted yet.")
        
        # Convert to numpy if needed
        if isinstance(X, Tensor):
            X = X.data
        X = np.asarray(X)
        
        n_samples = X.shape[0]
        decision_scores = np.zeros(n_samples)
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            predictions = self._predict_estimator(estimator, X)
            decision_scores += weight * predictions
        
        return decision_scores
    
    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Not used in AdaBoost as it uses weighted voting."""
        raise NotImplementedError("AdaBoost uses weighted voting logic")


class GradientBoostingClassifier(BaseEnsemble):
    """
    Gradient Boosting classifier.
    
    This implementation is simplified and focuses on the core concept
    of gradient boosting for educational purposes.
    
    Examples:
        >>> from fit.ensemble import GradientBoostingClassifier
        >>> 
        >>> # Create gradient boosting classifier
        >>> gb = GradientBoostingClassifier(
        ...     n_estimators=100,
        ...     learning_rate=0.1,
        ...     max_depth=3
        ... )
        >>> gb.fit(X_train, y_train)
        >>> predictions = gb.predict(X_test)
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, random_state: Optional[int] = None):
        """
        Initialize Gradient Boosting classifier.
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinks contribution of each tree
            max_depth: Maximum depth of individual regression estimators
            random_state: Random state for reproducibility
        """
        super().__init__(n_estimators=n_estimators, random_state=random_state)
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        # Simplified: we'll use the initial prediction as the mean
        self.init_prediction_ = None
    
    def _make_estimator(self) -> Any:
        """
        Create a new estimator instance (decision tree surrogate).
        
        Returns:
            New estimator instance
        """
        # For simplicity, use a small MLP as a surrogate for decision trees
        from fit.simple.models import MLP
        return MLP([1, 4, 1], activation="tanh")
    
    def _fit_estimator(self, estimator, X: np.ndarray, residuals: np.ndarray, sample_indices: np.ndarray):
        """
        Fit a single estimator to residuals.
        
        Args:
            estimator: The estimator to fit
            X: Training data
            residuals: Current residuals to fit
            sample_indices: Sample indices (not used in basic GB)
        """
        # Fit estimator to residuals
        if hasattr(estimator, 'fit'):
            estimator.fit(X, residuals)
    
    def _predict_estimator(self, estimator, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a single estimator.
        
        Args:
            estimator: The fitted estimator
            X: Input data
            
        Returns:
            Predictions from the estimator
        """
        if hasattr(estimator, 'predict'):
            return estimator.predict(X)
        elif hasattr(estimator, 'forward'):
            # For neural network models
            X_tensor = Tensor(X)
            predictions = estimator.forward(X_tensor)
            return predictions.data.flatten()
        else:
            raise ValueError(f"Estimator {estimator} has no predict or forward method")
    
    def fit(self, X: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor]) -> 'GradientBoostingClassifier':
        """
        Fit the gradient boosting model.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            Self for method chaining
        """
        # Convert to numpy if needed
        if isinstance(X, Tensor):
            X = X.data
        if isinstance(y, Tensor):
            y = y.data
            
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # Store classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            # Binary classification: convert to {0, 1}
            y_encoded = np.where(y == self.classes_[0], 0, 1)
        else:
            # Multi-class: use label encoding
            y_encoded = np.searchsorted(self.classes_, y)
        
        n_samples = X.shape[0]
        
        # Initialize with prior (class probability)
        if n_classes == 2:
            # Binary case: use log-odds
            pos_rate = np.mean(y_encoded)
            pos_rate = np.clip(pos_rate, 1e-15, 1 - 1e-15)  # Avoid log(0)
            self.init_prediction_ = np.log(pos_rate / (1 - pos_rate))
        else:
            # Multi-class: use most frequent class
            self.init_prediction_ = np.bincount(y_encoded).argmax()
        
        # Initialize predictions
        if n_classes == 2:
            current_predictions = np.full(n_samples, self.init_prediction_)
        else:
            current_predictions = np.full(n_samples, self.init_prediction_)
        
        # Clear previous estimators
        self.estimators_ = []
        
        # Fit estimators
        for stage in range(self.n_estimators):
            # Calculate residuals (negative gradient)
            if n_classes == 2:
                # Binary classification: logistic loss gradient
                probabilities = self._sigmoid(current_predictions)
                residuals = y_encoded - probabilities
            else:
                # Multi-class: simplified residuals
                residuals = y_encoded - current_predictions
            
            # Fit estimator to residuals
            estimator = self._make_estimator()
            self._fit_estimator(estimator, X, residuals, np.arange(n_samples))
            
            # Get predictions from the estimator
            tree_predictions = self._predict_estimator(estimator, X)
            
            # Update current predictions
            current_predictions += self.learning_rate * tree_predictions
            
            # Store the estimator
            self.estimators_.append(estimator)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Input data
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted_:
            raise ValueError("This GradientBoostingClassifier instance is not fitted yet.")
        
        # Get decision function values
        decision_values = self.decision_function(X)
        
        if len(self.classes_) == 2:
            # Binary classification
            predictions = (decision_values >= 0).astype(int)
            return self.classes_[predictions]
        else:
            # Multi-class
            predictions = np.round(decision_values).astype(int)
            predictions = np.clip(predictions, 0, len(self.classes_) - 1)
            return self.classes_[predictions]
    
    def decision_function(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Compute the decision function of X.
        
        Args:
            X: Input data
            
        Returns:
            Decision function values
        """
        if not self.is_fitted_:
            raise ValueError("This GradientBoostingClassifier instance is not fitted yet.")
        
        # Convert to numpy if needed
        if isinstance(X, Tensor):
            X = X.data
        X = np.asarray(X, dtype=np.float64)
        
        n_samples = X.shape[0]
        
        # Start with initial prediction
        predictions = np.full(n_samples, self.init_prediction_)
        
        # Add contributions from all estimators
        for estimator in self.estimators_:
            tree_predictions = self._predict_estimator(estimator, X)
            predictions += self.learning_rate * tree_predictions
        
        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Args:
            X: Input data
            
        Returns:
            Class probabilities
        """
        if len(self.classes_) != 2:
            raise NotImplementedError("predict_proba only implemented for binary classification")
        
        decision_values = self.decision_function(X)
        probabilities = self._sigmoid(decision_values)
        
        # Return probabilities for both classes
        proba = np.column_stack([1 - probabilities, probabilities])
        return proba
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid function with numerical stability.
        
        Args:
            x: Input values
            
        Returns:
            Sigmoid values
        """
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Not used in Gradient Boosting."""
        raise NotImplementedError("Gradient Boosting uses sequential fitting")


class SimpleBoostingClassifier(BaseEnsemble):
    """
    Simplified boosting classifier for educational purposes.
    
    This is a basic implementation that demonstrates the core
    concepts of boosting without the complexity of AdaBoost or Gradient Boosting.
    
    Examples:
        >>> from fit.ensemble import SimpleBoostingClassifier
        >>> from fit.simple.models import MLP
        >>> 
        >>> # Create simple boosting classifier
        >>> boost = SimpleBoostingClassifier(
        ...     base_estimator=MLP([4, 2]),
        ...     n_estimators=10
        ... )
        >>> boost.fit(X_train, y_train)
        >>> predictions = boost.predict(X_test)
    """
    
    def __init__(self, base_estimator=None, n_estimators: int = 10, 
                 random_state: Optional[int] = None):
        """
        Initialize simple boosting classifier.
        
        Args:
            base_estimator: Base estimator to boost
            n_estimators: Number of estimators
            random_state: Random state for reproducibility
        """
        super().__init__(n_estimators=n_estimators, random_state=random_state)
        self.base_estimator = base_estimator
        self.estimator_weights_ = []
    
    def _make_estimator(self) -> Any:
        """Create a new estimator instance."""
        if self.base_estimator is None:
            from fit.simple.models import MLP
            return MLP([1, 2], activation="tanh")
        
        if hasattr(self.base_estimator, 'copy'):
            return self.base_estimator.copy()
        else:
            estimator_class = self.base_estimator.__class__
            return estimator_class()
    
    def _fit_estimator(self, estimator, X: np.ndarray, y: np.ndarray, sample_indices: np.ndarray):
        """Fit estimator on bootstrap sample."""
        X_sample = X[sample_indices]
        y_sample = y[sample_indices]
        
        if hasattr(estimator, 'fit'):
            estimator.fit(X_sample, y_sample)
    
    def _predict_estimator(self, estimator, X: np.ndarray) -> np.ndarray:
        """Make predictions with estimator."""
        if hasattr(estimator, 'predict'):
            predictions = estimator.predict(X)
        elif hasattr(estimator, 'forward'):
            X_tensor = Tensor(X)
            predictions = estimator.forward(X_tensor).data
        else:
            raise ValueError(f"Estimator {estimator} has no predict or forward method")
        
        # Convert to class predictions if needed
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
        
        return predictions
    
    def fit(self, X: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor]) -> 'SimpleBoostingClassifier':
        """
        Fit the simple boosting model.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            Self for method chaining
        """
        # Convert to numpy if needed
        if isinstance(X, Tensor):
            X = X.data
        if isinstance(y, Tensor):
            y = y.data
            
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Clear previous results
        self.estimators_ = []
        self.estimator_weights_ = []
        
        # Fit estimators sequentially
        for i in range(self.n_estimators):
            # Create estimator
            estimator = self._make_estimator()
            
            # Generate bootstrap sample with emphasis on previously misclassified examples
            if i == 0:
                # First iteration: uniform sampling
                sample_indices = self._bootstrap_sample(X.shape[0])
            else:
                # Subsequent iterations: focus on misclassified examples
                prev_predictions = self._predict_estimator(self.estimators_[-1], X)
                misclassified = (prev_predictions != y)
                
                # Create weighted sampling favoring misclassified examples
                weights = np.ones(len(X))
                weights[misclassified] *= 2  # Double weight for misclassified
                weights /= weights.sum()
                
                sample_indices = np.random.choice(
                    len(X), size=len(X), replace=True, p=weights
                )
            
            # Fit estimator
            self._fit_estimator(estimator, X, y, sample_indices)
            
            # Calculate accuracy on full dataset
            predictions = self._predict_estimator(estimator, X)
            accuracy = np.mean(predictions == y)
            
            # Simple weight based on accuracy
            weight = max(0.1, accuracy)  # Minimum weight of 0.1
            
            # Store estimator and weight
            self.estimators_.append(estimator)
            self.estimator_weights_.append(weight)
        
        self.is_fitted_ = True
        return self
    
    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Combine predictions using weighted voting.
        
        Args:
            predictions: Array of shape (n_estimators, n_samples)
            
        Returns:
            Combined predictions
        """
        n_samples = predictions.shape[1]
        result = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Get weighted votes for this sample
            votes = predictions[:, i]
            weights = np.array(self.estimator_weights_)
            
            # Find unique classes and their weighted votes
            unique_classes = np.unique(votes)
            class_weights = np.zeros(len(unique_classes))
            
            for j, cls in enumerate(unique_classes):
                class_mask = (votes == cls)
                class_weights[j] = np.sum(weights[class_mask])
            
            # Choose class with highest weighted vote
            best_class_idx = np.argmax(class_weights)
            result[i] = unique_classes[best_class_idx]
        
        return result