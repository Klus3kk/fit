"""
Base classes for ensemble methods.

This module provides the foundation for building ensemble models
that combine multiple estimators to improve predictive performance.
"""

import numpy as np
from typing import List, Optional, Any, Union
from abc import ABC, abstractmethod

from fit.core.tensor import Tensor
from fit.nn.modules.base import Layer


class BaseEnsemble(Layer, ABC):
    """
    Base class for all ensemble methods.
    
    Warning: This class should not be used directly. Use derived classes instead.
    """
    
    def __init__(self, n_estimators: int = 10, random_state: Optional[int] = None):
        """
        Initialize base ensemble.
        
        Args:
            n_estimators: Number of estimators in the ensemble
            random_state: Random state for reproducibility
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.is_fitted_ = False
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def _fit_estimator(self, estimator, X: np.ndarray, y: np.ndarray, sample_indices: np.ndarray):
        """
        Fit a single estimator.
        
        Args:
            estimator: The estimator to fit
            X: Training data
            y: Target values
            sample_indices: Indices of samples to use for training
        """
        pass
    
    @abstractmethod
    def _predict_estimator(self, estimator, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a single estimator.
        
        Args:
            estimator: The fitted estimator
            X: Input data
            
        Returns:
            Predictions from the estimator
        """
        pass
    
    @abstractmethod
    def _make_estimator(self) -> Any:
        """
        Create a new estimator instance.
        
        Returns:
            New estimator instance
        """
        pass
    
    def fit(self, X: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor]) -> 'BaseEnsemble':
        """
        Fit the ensemble.
        
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
        
        # Clear previous estimators
        self.estimators_ = []
        
        # Fit each estimator
        for i in range(self.n_estimators):
            # Create new estimator
            estimator = self._make_estimator()
            
            # Generate bootstrap sample indices
            sample_indices = self._bootstrap_sample(X.shape[0])
            
            # Fit estimator
            self._fit_estimator(estimator, X, y, sample_indices)
            
            # Store fitted estimator
            self.estimators_.append(estimator)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Make predictions with the ensemble.
        
        Args:
            X: Input data
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted_:
            raise ValueError("This ensemble instance is not fitted yet.")
        
        # Convert to numpy if needed
        if isinstance(X, Tensor):
            X = X.data
        X = np.asarray(X)
        
        # Collect predictions from all estimators
        predictions = []
        for estimator in self.estimators_:
            pred = self._predict_estimator(estimator, X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Combine predictions
        return self._combine_predictions(predictions)
    
    @abstractmethod
    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Combine predictions from all estimators.
        
        Args:
            predictions: Array of shape (n_estimators, n_samples, ...)
            
        Returns:
            Combined predictions
        """
        pass
    
    def _bootstrap_sample(self, n_samples: int) -> np.ndarray:
        """
        Generate bootstrap sample indices.
        
        Args:
            n_samples: Number of samples in the dataset
            
        Returns:
            Bootstrap sample indices
        """
        return np.random.choice(n_samples, size=n_samples, replace=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for neural network compatibility.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        predictions = self.predict(x)
        return Tensor(predictions, requires_grad=x.requires_grad)


class VotingClassifier(BaseEnsemble):
    """
    Voting classifier for combining multiple classification models.
    
    Examples:
        >>> from fit.ensemble import VotingClassifier
        >>> from fit.simple.models import MLP
        >>> 
        >>> # Create base estimators
        >>> estimators = [
        ...     ('mlp1', MLP([4, 8, 3])),
        ...     ('mlp2', MLP([4, 16, 3])),
        ...     ('mlp3', MLP([4, 12, 3]))
        ... ]
        >>> 
        >>> # Create voting classifier
        >>> ensemble = VotingClassifier(estimators, voting='soft')
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """
    
    def __init__(self, estimators: List[tuple], voting: str = 'hard', weights: Optional[List[float]] = None):
        """
        Initialize voting classifier.
        
        Args:
            estimators: List of (name, estimator) tuples
            voting: Voting strategy ('hard' or 'soft')
            weights: Sequence of weights for the estimators
        """
        super().__init__(n_estimators=len(estimators))
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        
        if voting not in ['hard', 'soft']:
            raise ValueError("voting must be 'hard' or 'soft'")
    
    def _make_estimator(self) -> Any:
        """Not used in VotingClassifier as estimators are provided."""
        raise NotImplementedError("VotingClassifier uses provided estimators")
    
    def _fit_estimator(self, estimator, X: np.ndarray, y: np.ndarray, sample_indices: np.ndarray):
        """Not used in VotingClassifier as it uses all data."""
        raise NotImplementedError("VotingClassifier uses all training data")
    
    def _predict_estimator(self, estimator, X: np.ndarray) -> np.ndarray:
        """Make predictions with a single estimator."""
        if hasattr(estimator, 'predict'):
            return estimator.predict(X)
        elif hasattr(estimator, 'forward'):
            # For neural network models
            X_tensor = Tensor(X)
            predictions = estimator.forward(X_tensor)
            return predictions.data
        else:
            raise ValueError(f"Estimator {estimator} has no predict or forward method")
    
    def fit(self, X: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor]) -> 'VotingClassifier':
        """
        Fit all estimators.
        
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
        
        # Fit each estimator
        self.fitted_estimators_ = []
        
        for name, estimator in self.estimators:
            # Create a copy of the estimator
            if hasattr(estimator, 'copy'):
                fitted_est = estimator.copy()
            else:
                # For simple estimators, use the same instance
                fitted_est = estimator
            
            # Fit the estimator
            if hasattr(fitted_est, 'fit'):
                fitted_est.fit(X, y)
            
            self.fitted_estimators_.append(fitted_est)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Make predictions using voting.
        
        Args:
            X: Input data
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted_:
            raise ValueError("This VotingClassifier instance is not fitted yet.")
        
        # Convert to numpy if needed
        if isinstance(X, Tensor):
            X = X.data
        X = np.asarray(X)
        
        if self.voting == 'hard':
            # Hard voting: majority vote
            predictions = []
            for estimator in self.fitted_estimators_:
                pred = self._predict_estimator(estimator, X)
                # Convert probabilities to class predictions if needed
                if pred.ndim > 1 and pred.shape[1] > 1:
                    pred = np.argmax(pred, axis=1)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Apply weights if provided
            if self.weights is not None:
                # For hard voting with weights, we need a different approach
                # For simplicity, we'll ignore weights in hard voting
                pass
            
            # Majority vote
            return self._majority_vote(predictions)
        
        else:  # soft voting
            # Soft voting: average probabilities
            predictions = []
            for estimator in self.fitted_estimators_:
                pred = self._predict_estimator(estimator, X)
                # Ensure predictions are probabilities
                if pred.ndim == 1 or pred.shape[1] == 1:
                    # Convert binary predictions to probability format
                    prob = np.zeros((len(pred), 2))
                    prob[:, 1] = pred.flatten()
                    prob[:, 0] = 1 - pred.flatten()
                    pred = prob
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Apply weights
            if self.weights is not None:
                weights = np.array(self.weights).reshape(-1, 1, 1)
                predictions = predictions * weights
            
            # Average probabilities
            avg_probs = np.mean(predictions, axis=0)
            return np.argmax(avg_probs, axis=1)
    
    def _majority_vote(self, predictions: np.ndarray) -> np.ndarray:
        """
        Compute majority vote from predictions.
        
        Args:
            predictions: Array of shape (n_estimators, n_samples)
            
        Returns:
            Majority vote predictions
        """
        n_samples = predictions.shape[1]
        result = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Get votes for this sample
            votes = predictions[:, i]
            
            # Count votes
            unique_votes, counts = np.unique(votes, return_counts=True)
            
            # Get majority vote
            majority_idx = np.argmax(counts)
            result[i] = unique_votes[majority_idx]
        
        return result
    
    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Not used in VotingClassifier."""
        raise NotImplementedError("VotingClassifier uses custom prediction logic")


class BaggingClassifier(BaseEnsemble):
    """
    Bagging classifier that fits multiple models on bootstrap samples.
    
    Examples:
        >>> from fit.ensemble import BaggingClassifier
        >>> from fit.simple.models import MLP
        >>> 
        >>> # Create bagging classifier
        >>> ensemble = BaggingClassifier(
        ...     base_estimator=MLP([4, 8, 3]),
        ...     n_estimators=10,
        ...     random_state=42
        ... )
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """
    
    def __init__(self, base_estimator=None, n_estimators: int = 10, 
                 max_samples: Union[int, float] = 1.0, 
                 random_state: Optional[int] = None):
        """
        Initialize bagging classifier.
        
        Args:
            base_estimator: Base estimator to fit on bootstrap samples
            n_estimators: Number of estimators in the ensemble
            max_samples: Number/fraction of samples to draw for each estimator
            random_state: Random state for reproducibility
        """
        super().__init__(n_estimators=n_estimators, random_state=random_state)
        self.base_estimator = base_estimator
        self.max_samples = max_samples
    
    def _make_estimator(self) -> Any:
        """
        Create a new estimator instance.
        
        Returns:
            New estimator instance
        """
        if self.base_estimator is None:
            # Default to a simple MLP
            from fit.simple.models import MLP
            return MLP([4, 8, 3])  # This would need to be adapted based on data
        
        # Create a copy of the base estimator
        if hasattr(self.base_estimator, 'copy'):
            return self.base_estimator.copy()
        else:
            # For simple cases, create a new instance
            estimator_class = self.base_estimator.__class__
            return estimator_class()
    
    def _fit_estimator(self, estimator, X: np.ndarray, y: np.ndarray, sample_indices: np.ndarray):
        """
        Fit a single estimator on bootstrap sample.
        
        Args:
            estimator: The estimator to fit
            X: Training data
            y: Target values
            sample_indices: Bootstrap sample indices
        """
        # Get bootstrap sample
        X_bootstrap = X[sample_indices]
        y_bootstrap = y[sample_indices]
        
        # Fit estimator
        if hasattr(estimator, 'fit'):
            estimator.fit(X_bootstrap, y_bootstrap)
    
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
            return predictions.data
        else:
            raise ValueError(f"Estimator {estimator} has no predict or forward method")
    
    def _combine_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Combine predictions using majority voting.
        
        Args:
            predictions: Array of shape (n_estimators, n_samples, ...)
            
        Returns:
            Combined predictions
        """
        # For classification, use majority voting
        if predictions.ndim == 2:
            # Binary/multi-class predictions
            return self._majority_vote(predictions)
        else:
            # Probability predictions
            return np.argmax(np.mean(predictions, axis=0), axis=1)
    
    def _majority_vote(self, predictions: np.ndarray) -> np.ndarray:
        """Compute majority vote."""
        n_samples = predictions.shape[1]
        result = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            votes = predictions[:, i]
            unique_votes, counts = np.unique(votes, return_counts=True)
            majority_idx = np.argmax(counts)
            result[i] = unique_votes[majority_idx]
        
        return result
    
    def _bootstrap_sample(self, n_samples: int) -> np.ndarray:
        """
        Generate bootstrap sample indices with max_samples control.
        
        Args:
            n_samples: Number of samples in the dataset
            
        Returns:
            Bootstrap sample indices
        """
        if isinstance(self.max_samples, float):
            sample_size = int(n_samples * self.max_samples)
        else:
            sample_size = min(self.max_samples, n_samples)
        
        return np.random.choice(n_samples, size=sample_size, replace=True)