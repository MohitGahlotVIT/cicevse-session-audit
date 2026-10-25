import numpy as np

from src.evaluation.metrics import compute_metrics


def test_perfect_predictions():
    labels = ["a", "b", "c"]
    result = compute_metrics(np.array(labels), np.array(labels), labels)
    assert result["macro_f1"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["n_classes_evaluated"] == 3
    assert result["absent_classes"] == []


def test_absent_class_excluded():
    labels = ["a", "b", "c"]
    y_true = np.array(["a", "b", "a", "b"])
    result = compute_metrics(y_true, y_true, labels)
    assert result["n_classes_evaluated"] == 2
    assert result["absent_classes"] == ["c"]
    assert result["macro_f1"] == 1.0
