"""
Classification metrics for model evaluation.

This module provides comprehensive metrics for evaluating classification models,
including binary and multi-class scenarios.
"""

import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple
import warnings


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Classification accuracy score.
    
    Args:
        y_true: Ground truth (correct) labels
        y_pred: Predicted labels
        normalize: If True, return fraction of correctly classified samples
        
    Returns:
        Accuracy score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    correct = np.sum(y_true == y_pred)
    
    if normalize:
        return correct / len(y_true)
    else:
        return correct


def precision_score(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    average: str = 'binary',
    pos_label: int = 1,
    zero_division: Union[str, int] = 'warn'
) -> Union[float, np.ndarray]:
    """
    Compute the precision score.
    
    Precision = TP / (TP + FP)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted', None)
        pos_label: Label of the positive class (for binary classification)
        zero_division: Value to return when there is a zero division
        
    Returns:
        Precision score(s)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if average == 'binary':
        return _binary_precision(y_true, y_pred, pos_label, zero_division)
    else:
        return _multiclass_precision(y_true, y_pred, average, zero_division)


def recall_score(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    average: str = 'binary',
    pos_label: int = 1,
    zero_division: Union[str, int] = 'warn'
) -> Union[float, np.ndarray]:
    """
    Compute the recall score.
    
    Recall = TP / (TP + FN)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted', None)
        pos_label: Label of the positive class (for binary classification)
        zero_division: Value to return when there is a zero division
        
    Returns:
        Recall score(s)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if average == 'binary':
        return _binary_recall(y_true, y_pred, pos_label, zero_division)
    else:
        return _multiclass_recall(y_true, y_pred, average, zero_division)


def f1_score(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    average: str = 'binary',
    pos_label: int = 1,
    zero_division: Union[str, int] = 'warn'
) -> Union[float, np.ndarray]:
    """
    Compute the F1 score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted', None)
        pos_label: Label of the positive class (for binary classification)
        zero_division: Value to return when there is a zero division
        
    Returns:
        F1 score(s)
    """
    precision = precision_score(y_true, y_pred, average, pos_label, zero_division)
    recall = recall_score(y_true, y_pred, average, pos_label, zero_division)
    
    if isinstance(precision, np.ndarray):
        # Handle array case
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * (precision * recall) / (precision + recall)
            f1 = np.where((precision + recall) == 0, 0, f1)
        return f1
    else:
        # Handle scalar case
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


def fbeta_score(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    beta: float = 1.0,
    average: str = 'binary',
    pos_label: int = 1,
    zero_division: Union[str, int] = 'warn'
) -> Union[float, np.ndarray]:
    """
    Compute the F-beta score.
    
    F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        beta: Weight of recall in harmonic mean
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted', None)
        pos_label: Label of the positive class (for binary classification)
        zero_division: Value to return when there is a zero division
        
    Returns:
        F-beta score(s)
    """
    precision = precision_score(y_true, y_pred, average, pos_label, zero_division)
    recall = recall_score(y_true, y_pred, average, pos_label, zero_division)
    
    beta_squared = beta ** 2
    
    if isinstance(precision, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
            fbeta = np.where((beta_squared * precision + recall) == 0, 0, fbeta)
        return fbeta
    else:
        if beta_squared * precision + recall == 0:
            return 0.0
        return (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)


def confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix to evaluate classification accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels to index the matrix
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        
    Returns:
        Confusion matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    
    n_labels = len(labels)
    label_to_ind = {label: i for i, label in enumerate(labels)}
    
    # Create confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_ind and pred_label in label_to_ind:
            true_idx = label_to_ind[true_label]
            pred_idx = label_to_ind[pred_label]
            cm[true_idx, pred_idx] += 1
    
    # Normalize if requested
    if normalize == 'true':
        # Normalize over true labels (rows)
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums!=0)
    elif normalize == 'pred':
        # Normalize over predicted labels (columns)
        cm = cm.astype(float)
        col_sums = cm.sum(axis=0, keepdims=True)
        cm = np.divide(cm, col_sums, out=np.zeros_like(cm), where=col_sums!=0)
    elif normalize == 'all':
        # Normalize over all samples
        cm = cm.astype(float)
        total = cm.sum()
        if total > 0:
            cm = cm / total
    
    return cm


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    digits: int = 2,
    zero_division: Union[str, int] = 'warn'
) -> str:
    """
    Build a text report showing the main classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels to include in the report
        target_names: Display names for the labels
        digits: Number of digits for formatting
        zero_division: Value to return when there is a zero division
        
    Returns:
        Text summary of classification metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    if target_names is None:
        target_names = [str(label) for label in labels]
    elif len(target_names) != len(labels):
        raise ValueError("Number of target names must equal number of labels")
    
    # Calculate metrics for each class
    report_lines = []
    report_lines.append(f"{'':>12} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}")
    report_lines.append("")
    
    # Per-class metrics
    precisions = precision_score(y_true, y_pred, average=None, zero_division=zero_division)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=zero_division)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=zero_division)
    
    supports = []
    for label in labels:
        support = np.sum(y_true == label)
        supports.append(support)
    
    for i, (name, precision, recall, f1, support) in enumerate(zip(target_names, precisions, recalls, f1s, supports)):
        report_lines.append(
            f"{name:>12} {precision:>9.{digits}f} {recall:>9.{digits}f} "
            f"{f1:>9.{digits}f} {support:>9}"
        )
    
    report_lines.append("")
    
    # Averages
    total_support = np.sum(supports)
    
    # Macro averages
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    
    report_lines.append(
        f"{'macro avg':>12} {macro_precision:>9.{digits}f} {macro_recall:>9.{digits}f} "
        f"{macro_f1:>9.{digits}f} {total_support:>9}"
    )
    
    # Weighted averages
    weighted_precision = np.average(precisions, weights=supports)
    weighted_recall = np.average(recalls, weights=supports)
    weighted_f1 = np.average(f1s, weights=supports)
    
    report_lines.append(
        f"{'weighted avg':>12} {weighted_precision:>9.{digits}f} {weighted_recall:>9.{digits}f} "
        f"{weighted_f1:>9.{digits}f} {total_support:>9}"
    )
    
    return "\n".join(report_lines)


def roc_auc_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    average: str = 'macro',
    multi_class: str = 'raise'
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    
    Args:
        y_true: Ground truth labels
        y_score: Target scores (probabilities of positive class or decision function)
        average: Averaging strategy for multiclass ('macro', 'weighted', 'micro')
        multi_class: Strategy for multiclass ('raise', 'ovr', 'ovo')
        
    Returns:
        ROC AUC score
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    classes = np.unique(y_true)
    
    if len(classes) == 2:
        # Binary classification
        return _binary_roc_auc(y_true, y_score, classes)
    else:
        if multi_class == 'raise':
            raise ValueError("ROC AUC is not defined for multiclass without specifying multi_class parameter")
        elif multi_class == 'ovr':
            return _multiclass_roc_auc_ovr(y_true, y_score, classes, average)
        else:
            raise NotImplementedError("OvO multiclass ROC AUC not implemented yet")


def log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
    normalize: bool = True
) -> float:
    """
    Logistic regression loss (cross-entropy loss).
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        eps: Small value to clip probabilities
        normalize: If True, return the mean loss per sample
        
    Returns:
        Log loss
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Clip probabilities to prevent log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    if y_pred.ndim == 1:
        # Binary case
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        # Multiclass case
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
    
    if normalize:
        return np.mean(loss)
    else:
        return np.sum(loss)


def cohen_kappa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Cohen's kappa: a statistic that measures inter-annotator agreement.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Cohen's kappa coefficient
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    n_observed = np.sum(cm)
    
    # Observed agreement
    po = np.trace(cm) / n_observed
    
    # Expected agreement
    marginal_true = np.sum(cm, axis=1) / n_observed
    marginal_pred = np.sum(cm, axis=0) / n_observed
    pe = np.sum(marginal_true * marginal_pred)
    
    # Cohen's kappa
    if pe == 1:
        return 1.0
    else:
        return (po - pe) / (1 - pe)


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Matthews correlation coefficient (MCC).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Matthews correlation coefficient
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        # Binary case
        tn, fp, fn, tp = cm.ravel()
        
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        else:
            return numerator / denominator
    else:
        # Multiclass case
        n_samples = np.sum(cm)
        cov_ytyp = np.trace(cm) * n_samples - np.sum(cm.sum(axis=1) * cm.sum(axis=0))
        cov_ypyp = n_samples**2 - np.sum(cm.sum(axis=0)**2)
        cov_ytyt = n_samples**2 - np.sum(cm.sum(axis=1)**2)
        
        if cov_ypyp * cov_ytyt == 0:
            return 0.0
        else:
            return cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)


# Helper functions
def _binary_precision(y_true, y_pred, pos_label, zero_division):
    """Calculate precision for binary classification."""
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    
    if tp + fp == 0:
        if zero_division == 'warn':
            warnings.warn("F-score is ill-defined and being set to 0.0 due to no predicted samples.")
            return 0.0
        else:
            return zero_division
    
    return tp / (tp + fp)


def _binary_recall(y_true, y_pred, pos_label, zero_division):
    """Calculate recall for binary classification."""
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    
    if tp + fn == 0:
        if zero_division == 'warn':
            warnings.warn("F-score is ill-defined and being set to 0.0 due to no true samples.")
            return 0.0
        else:
            return zero_division
    
    return tp / (tp + fn)


def _multiclass_precision(y_true, y_pred, average, zero_division):
    """Calculate precision for multiclass classification."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    
    for label in labels:
        precision = _binary_precision(y_true, y_pred, label, zero_division)
        precisions.append(precision)
    
    precisions = np.array(precisions)
    
    if average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        tp_total = np.sum([np.sum((y_true == label) & (y_pred == label)) for label in labels])
        fp_total = np.sum([np.sum((y_true != label) & (y_pred == label)) for label in labels])
        if tp_total + fp_total == 0:
            return 0.0
        return tp_total / (tp_total + fp_total)
    elif average == 'weighted':
        supports = [np.sum(y_true == label) for label in labels]
        return np.average(precisions, weights=supports)
    elif average is None:
        return precisions
    else:
        raise ValueError(f"Unknown average method: {average}")


def _multiclass_recall(y_true, y_pred, average, zero_division):
    """Calculate recall for multiclass classification."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    
    for label in labels:
        recall = _binary_recall(y_true, y_pred, label, zero_division)
        recalls.append(recall)
    
    recalls = np.array(recalls)
    
    if average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        tp_total = np.sum([np.sum((y_true == label) & (y_pred == label)) for label in labels])
        fn_total = np.sum([np.sum((y_true == label) & (y_pred != label)) for label in labels])
        if tp_total + fn_total == 0:
            return 0.0
        return tp_total / (tp_total + fn_total)
    elif average == 'weighted':
        supports = [np.sum(y_true == label) for label in labels]
        return np.average(recalls, weights=supports)
    elif average is None:
        return recalls
    else:
        raise ValueError(f"Unknown average method: {average}")


def _binary_roc_auc(y_true, y_score, classes):
    """Calculate ROC AUC for binary classification."""
    pos_label = classes[1]
    
    # Convert to binary (0, 1)
    y_true_binary = (y_true == pos_label).astype(int)
    
    # Sort by score
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true_binary[desc_score_indices]
    
    # Calculate cumulative counts
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]
    
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    # Calculate ROC curve
    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, false positive value should be meaningless")
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]
    
    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, true positive value should be meaningless")
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]
    
    # Calculate AUC using trapezoidal rule
    return np.trapz(tpr, fpr)


def _multiclass_roc_auc_ovr(y_true, y_score, classes, average):
    """Calculate ROC AUC for multiclass using One-vs-Rest approach."""
    aucs = []
    
    for i, class_label in enumerate(classes):
        # Create binary problem: current class vs all others
        y_binary = (y_true == class_label).astype(int)
        y_score_binary = y_score[:, i] if y_score.ndim > 1 else y_score
        
        try:
            auc = _binary_roc_auc(y_binary, y_score_binary, [0, 1])
            aucs.append(auc)
        except:
            aucs.append(0.5)  # Random classifier
    
    aucs = np.array(aucs)
    
    if average == 'macro':
        return np.mean(aucs)
    elif average == 'weighted':
        supports = [np.sum(y_true == label) for label in classes]
        return np.average(aucs, weights=supports)
    else:
        return aucs


def get_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    pos_label: int = 1,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Get a comprehensive set of classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        pos_label: Positive label for binary classification
        average: Averaging strategy for multiclass metrics
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, pos_label=pos_label)
    
    # Additional metrics
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    
    # Probability-based metrics (if probabilities provided)
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if y_pred_proba.ndim == 1:
                    proba_pos = y_pred_proba
                else:
                    proba_pos = y_pred_proba[:, 1]
                
                metrics['roc_auc'] = roc_auc_score(y_true, proba_pos)
                metrics['log_loss'] = log_loss(y_true, proba_pos)
            else:
                # Multiclass
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                 average=average, multi_class='ovr')
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except Exception as e:
            warnings.warn(f"Could not calculate probability-based metrics: {e}")
    
    return metrics