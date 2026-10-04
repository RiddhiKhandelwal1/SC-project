"""
Evaluation Metrics — Classification metrics, confusion matrix,
and per-class reporting.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    idx_to_class : dict or None
        Mapping from index to class name for display.

    Returns
    -------
    dict with: accuracy, precision, recall, f1, macro_f1, weighted_f1,
               confusion_matrix, classification_report, per_class_metrics
    """
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Aggregated
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    # Target names
    unique_labels = sorted(set(y_true) | set(y_pred))
    if idx_to_class:
        target_names = [idx_to_class.get(i, str(i)) for i in unique_labels]
    else:
        target_names = [str(i) for i in unique_labels]

    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
    )

    # Per-class metrics dict
    per_class = {}
    for i, label_idx in enumerate(unique_labels):
        name = idx_to_class.get(label_idx, str(label_idx)) if idx_to_class else str(label_idx)
        if i < len(precision):
            per_class[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(np.sum(y_true == label_idx)),
            }

    # Dataset size warning
    warnings = []
    total = len(y_true)
    if total < 100:
        warnings.append(
            f"⚠️ Very small evaluation set ({total} samples). "
            f"Metrics may not be reliable."
        )
    if total < 30 * len(unique_labels):
        warnings.append(
            f"⚠️ Only ~{total // max(1, len(unique_labels))} samples per class on average. "
            f"Consider using more data for reliable evaluation."
        )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm,
        "classification_report": report,
        "per_class_metrics": per_class,
        "warnings": warnings,
    }


def get_predictions(model, wav_data, spec_data) -> np.ndarray:
    """
    Get predicted class indices from the model.

    Parameters
    ----------
    model : tf.keras.Model
    wav_data : np.ndarray
    spec_data : np.ndarray

    Returns
    -------
    np.ndarray of predicted class indices
    """
    probs = model.predict([wav_data, spec_data], verbose=0)
    return np.argmax(probs, axis=-1)


def get_probabilities(model, wav_data, spec_data) -> np.ndarray:
    """
    Get class probability distributions from the model.

    Returns
    -------
    np.ndarray of shape (N, num_classes)
    """
    return model.predict([wav_data, spec_data], verbose=0)
