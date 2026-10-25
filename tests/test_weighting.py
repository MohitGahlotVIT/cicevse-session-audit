import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


def test_balanced_weights_equal_class_totals():
    labels = np.array(["a"] * 8 + ["b"] * 2 + ["c"] * 1)
    weights = compute_sample_weight("balanced", labels)
    totals = {
        label: float(weights[labels == label].sum())
        for label in ("a", "b", "c")
    }
    assert np.isclose(totals["a"], totals["b"])
    assert np.isclose(totals["b"], totals["c"])


def test_weights_are_finite_and_positive():
    labels = np.array(["a"] * 8 + ["b"] * 2 + ["c"] * 1)
    weights = compute_sample_weight("balanced", labels)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()


def test_inverse_frequency_formula():
    labels = np.array(["a"] * 8 + ["b"] * 2 + ["c"] * 1)
    weights = compute_sample_weight("balanced", labels)
    n_samples = len(labels)
    n_classes = 3
    assert np.isclose(weights[0], n_samples / (n_classes * 8))
    assert np.isclose(weights[8], n_samples / (n_classes * 2))
    assert np.isclose(weights[-1], n_samples / (n_classes * 1))
