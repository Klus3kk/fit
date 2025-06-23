"""
Regression metrics for model evaluation.

This module provides comprehensive metrics for evaluating regression models.
"""

import numpy as np
from typing import Union, Optional, Dict, Any
import warnings


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, squared: bool = True) -> float:
    """
    Mean squared error regression loss.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        squared: If True returns MSE, if False returns RMSE
        
    Returns:
        MSE or RMSE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    
    if squared:
        return mse
    else:
        return np.sqrt(mse)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute error regression loss.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        MAE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return np.mean(np.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute percentage error regression loss.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        MAPE value as percentage
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Avoid division by zero
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (coefficient of determination) regression score.
    
    R² = 1 - SS_res / SS_tot
    where SS_res = Σ(y_true - y_pred)² and SS_tot = Σ(y_true - y_mean)²
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        R² score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        # All y_true values are the same
        if ss_res == 0:
            return 1.0  # Perfect prediction
        else:
            return 0.0  # Constant prediction but not perfect
    
    return 1 - (ss_res / ss_tot)


def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Adjusted R² score that accounts for the number of features.
    
    Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    where n is the number of samples and p is the number of features.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        n_features: Number of features used in the model
        
    Returns:
        Adjusted R² score
    """
    r2 = r2_score(y_true, y_pred)
    n_samples = len(y_true)
    
    if n_samples <= n_features + 1:
        warnings.warn("Number of samples should be greater than number of features + 1")
        return r2
    
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return adjusted_r2


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Median absolute error regression loss.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        Median absolute error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return np.median(np.abs(y_true - y_pred))


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Maximum residual error.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        Maximum absolute error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return np.max(np.abs(y_true - y_pred))


def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Explained variance regression score.
    
    Explained Variance = 1 - Var(y_true - y_pred) / Var(y_true)
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        Explained variance score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    var_y = np.var(y_true)
    if var_y == 0:
        return 1.0 if np.var(y_true - y_pred) == 0 else 0.0
    
    return 1 - np.var(y_true - y_pred) / var_y


def mean_squared_log_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared logarithmic error regression loss.
    
    MSLE = mean((log(1 + y_true) - log(1 + y_pred))²)
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        MSLE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("MSLE cannot be used when targets contain negative values")
    
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)


def mean_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Poisson deviance regression loss.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        Mean Poisson deviance
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if np.any(y_pred <= 0):
        raise ValueError("Poisson deviance requires positive predicted values")
    
    return 2 * np.mean(y_pred - y_true * np.log(y_pred))


def mean_gamma_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Gamma deviance regression loss.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        Mean Gamma deviance
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if np.any(y_true <= 0) or np.any(y_pred <= 0):
        raise ValueError("Gamma deviance requires positive values")
    
    return 2 * np.mean(np.log(y_pred / y_true) + y_true / y_pred - 1)


def mean_tweedie_deviance(y_true: np.ndarray, y_pred: np.ndarray, power: float = 0) -> float:
    """
    Mean Tweedie deviance regression loss.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        power: Tweedie power parameter
        
    Returns:
        Mean Tweedie deviance
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if power < 0:
        raise ValueError("Tweedie power must be non-negative")
    
    if power == 0:
        # Normal distribution (MSE)
        return mean_squared_error(y_true, y_pred)
    elif power == 1:
        # Poisson distribution
        return mean_poisson_deviance(y_true, y_pred)
    elif power == 2:
        # Gamma distribution
        return mean_gamma_deviance(y_true, y_pred)
    else:
        # General Tweedie
        if np.any(y_pred <= 0):
            raise ValueError("Tweedie deviance requires positive predicted values")
        
        if power == 1:
            dev = y_pred - y_true * np.log(y_pred)
        elif power == 2:
            dev = np.log(y_pred) + y_true / y_pred
        else:
            dev = (y_true ** (2 - power) / ((1 - power) * (2 - power)) - 
                   y_true * y_pred ** (1 - power) / (1 - power) + 
                   y_pred ** (2 - power) / (2 - power))
        
        return 2 * np.mean(dev)


def d2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    D² score, the fraction of deviance explained.
    
    D² = 1 - deviance(y_true, y_pred) / deviance(y_true, y_mean)
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        D² score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    y_mean = np.mean(y_true)
    
    # Use MSE as the deviance measure
    dev_pred = np.sum((y_true - y_pred) ** 2)
    dev_mean = np.sum((y_true - y_mean) ** 2)
    
    if dev_mean == 0:
        return 1.0 if dev_pred == 0 else 0.0
    
    return 1 - dev_pred / dev_mean


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric mean absolute percentage error (SMAPE).
    
    SMAPE = 100 * mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|))
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        SMAPE value as percentage
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1, denominator)
    
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / denominator)


def mean_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean directional accuracy for time series forecasting.
    
    Measures the percentage of times the predicted direction of change
    matches the actual direction of change.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        Directional accuracy as percentage
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) < 2:
        raise ValueError("Need at least 2 observations for directional accuracy")
    
    # Calculate direction of change
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    # Count correct directions
    correct_directions = np.sum(true_direction == pred_direction)
    total_directions = len(true_direction)
    
    return 100 * correct_directions / total_directions


def mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean bias error (average residual).
    
    MBE = mean(y_pred - y_true)
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        Mean bias error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return np.mean(y_pred - y_true)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root mean squared error.
    
    RMSE = sqrt(mean((y_true - y_pred)²))
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        RMSE value
    """
    return mean_squared_error(y_true, y_pred, squared=False)


def normalized_root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, normalization: str = 'range') -> float:
    """
    Normalized root mean squared error.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        normalization: Normalization method ('range', 'mean', 'std')
        
    Returns:
        Normalized RMSE value
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    
    if normalization == 'range':
        norm_factor = np.max(y_true) - np.min(y_true)
    elif normalization == 'mean':
        norm_factor = np.mean(y_true)
    elif normalization == 'std':
        norm_factor = np.std(y_true)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    if norm_factor == 0:
        warnings.warn("Normalization factor is zero, returning RMSE")
        return rmse
    
    return rmse / norm_factor


def coefficient_of_variation_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Variation of RMSE.
    
    CV(RMSE) = RMSE / mean(y_true)
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        CV(RMSE) value as percentage
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    mean_true = np.mean(y_true)
    
    if mean_true == 0:
        warnings.warn("Mean of y_true is zero, CV(RMSE) is undefined")
        return float('inf')
    
    return 100 * rmse / abs(mean_true)


def pearson_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson correlation coefficient between true and predicted values.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        Pearson correlation coefficient
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return np.corrcoef(y_true, y_pred)[0, 1]


def spearman_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman rank correlation coefficient between true and predicted values.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        Spearman correlation coefficient
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate ranks
    rank_true = np.argsort(np.argsort(y_true))
    rank_pred = np.argsort(np.argsort(y_pred))
    
    # Calculate Spearman correlation
    return np.corrcoef(rank_true, rank_pred)[0, 1]


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Concordance correlation coefficient (Lin's CCC).
    
    CCC combines precision and accuracy to measure agreement.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        Concordance correlation coefficient
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    # Calculate variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    # Calculate covariance
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    # Calculate CCC
    numerator = 2 * cov
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    
    return numerator / denominator


def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error (MASE) for time series forecasting.
    
    MASE = MAE / MAE_naive
    where MAE_naive is the MAE of a naive forecast on training data.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        y_train: Training data for calculating naive forecast error
        
    Returns:
        MASE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)
    
    # Calculate MAE of predictions
    mae_pred = mean_absolute_error(y_true, y_pred)
    
    # Calculate naive forecast MAE on training data (seasonal naive with period=1)
    if len(y_train) < 2:
        raise ValueError("Need at least 2 training observations for MASE")
    
    naive_forecast = y_train[:-1]  # Use previous value as forecast
    mae_naive = mean_absolute_error(y_train[1:], naive_forecast)
    
    if mae_naive == 0:
        warnings.warn("Naive forecast MAE is zero, MASE is undefined")
        return float('inf') if mae_pred > 0 else 0.0
    
    return mae_pred / mae_naive


def get_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multioutput: str = 'uniform_average'
) -> Dict[str, float]:
    """
    Get a comprehensive set of regression metrics.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        multioutput: How to handle multioutput ('uniform_average', 'raw_values')
        
    Returns:
        Dictionary of metric names and values
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = root_mean_squared_error(y_true, y_pred)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Additional metrics
    try:
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    except:
        pass
    
    try:
        metrics['smape'] = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    except:
        pass
    
    metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
    metrics['max_error'] = max_error(y_true, y_pred)
    metrics['median_absolute_error'] = median_absolute_error(y_true, y_pred)
    metrics['mean_bias_error'] = mean_bias_error(y_true, y_pred)
    
    # Correlation metrics
    try:
        metrics['pearson_corr'] = pearson_correlation_coefficient(y_true, y_pred)
        metrics['spearman_corr'] = spearman_correlation_coefficient(y_true, y_pred)
        metrics['concordance_corr'] = concordance_correlation_coefficient(y_true, y_pred)
    except:
        pass
    
    # Time series specific metrics
    if len(y_true) >= 2:
        try:
            metrics['directional_accuracy'] = mean_directional_accuracy(y_true, y_pred)
        except:
            pass
    
    return metrics


def regression_error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Comprehensive error analysis for regression models.
    
    Args:
        y_true: Ground truth target values
        y_pred: Estimated target values
        
    Returns:
        Dictionary with error analysis results
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    residuals = y_true - y_pred
    
    analysis = {}
    
    # Basic statistics
    analysis['residual_mean'] = np.mean(residuals)
    analysis['residual_std'] = np.std(residuals)
    analysis['residual_min'] = np.min(residuals)
    analysis['residual_max'] = np.max(residuals)
    analysis['residual_median'] = np.median(residuals)
    
    # Percentiles
    analysis['residual_q25'] = np.percentile(residuals, 25)
    analysis['residual_q75'] = np.percentile(residuals, 75)
    analysis['residual_iqr'] = analysis['residual_q75'] - analysis['residual_q25']
    
    # Skewness and kurtosis (simplified)
    analysis['residual_skewness'] = np.mean(((residuals - np.mean(residuals)) / np.std(residuals)) ** 3)
    analysis['residual_kurtosis'] = np.mean(((residuals - np.mean(residuals)) / np.std(residuals)) ** 4) - 3
    
    # Error distribution
    analysis['positive_errors'] = np.sum(residuals > 0)
    analysis['negative_errors'] = np.sum(residuals < 0)
    analysis['zero_errors'] = np.sum(residuals == 0)
    
    # Large errors
    large_error_threshold = 2 * np.std(residuals)
    analysis['large_errors'] = np.sum(np.abs(residuals) > large_error_threshold)
    analysis['large_error_percentage'] = 100 * analysis['large_errors'] / len(residuals)
    
    return analysis