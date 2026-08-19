"""Classification metrics matching Section 3.6, Eqs. 12-16:
accuracy, sensitivity, specificity, F1, NPV, plus ROC-AUC.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Args:
        y_true: ground-truth binary labels (0=non-stone, 1=stone)
        y_pred: predicted binary labels (argmax of softmax)
        y_prob: predicted probability of the positive (stone) class
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0        # Eq. 13
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0        # Eq. 14
    f1 = f1_score(y_true, y_pred, zero_division=0)                # Eq. 15
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0                # Eq. 16

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # e.g. only one class present in a tiny batch

    return {
        "accuracy": accuracy,          # Eq. 12
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "npv": npv,
        "auc": auc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
