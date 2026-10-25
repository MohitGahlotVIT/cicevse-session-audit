"""Classification metrics with explicit handling of absent classes."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """Compute per-class and aggregate metrics for a fixed label set."""
    present = [label for label in labels if label in set(y_true) or label in set(y_pred)]
    absent = [label for label in labels if label not in present]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(present)
    }
    macro_f1 = float(np.mean([per_class[label]["f1"] for label in present]))
    macro_precision = float(np.mean([per_class[label]["precision"] for label in present]))
    macro_recall = float(np.mean([per_class[label]["recall"] for label in present]))
    matrix = confusion_matrix(y_true, y_pred, labels=present)
    row_totals = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        row_totals,
        out=np.zeros(len(present), dtype=float),
        where=row_totals > 0,
    )
    balanced_accuracy = float(recalls.mean())
    accuracy = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    weighted_f1 = float(
        np.average(
            [per_class[label]["f1"] for label in present],
            weights=[per_class[label]["support"] for label in present],
        )
    )
    return {
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "n_classes_evaluated": len(present),
        "evaluated_classes": present,
        "absent_classes": absent,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": present,
    }
