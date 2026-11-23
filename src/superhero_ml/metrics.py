from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from .config import CLASS_LABELS


def classification_metrics(y_true, y_pred, y_probability=None) -> dict[str, Any]:
    precision = precision_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "hero_precision": float(precision[0]),
        "hero_recall": float(recall[0]),
        "hero_f1": float(f1[0]),
        "villain_precision": float(precision[1]),
        "villain_recall": float(recall[1]),
        "villain_f1": float(f1[1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
    }
    if y_probability is not None:
        result["log_loss"] = float(log_loss(y_true, np.asarray(y_probability), labels=CLASS_LABELS))
    return result

