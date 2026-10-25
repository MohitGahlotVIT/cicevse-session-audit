import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_committed_run_matrix_is_complete():
    matrix = pd.read_csv(ROOT / "configs" / "run_matrix.csv")
    assert len(matrix) == 40
    expected = {
        (model, experiment, protocol, weighting, seed)
        for model in ("xgboost", "lightgbm")
        for experiment, protocol, weighting in (
            ("E1", "random", "standard"),
            ("E2", "random", "inverse_frequency"),
            ("E3", "session", "standard"),
            ("E4", "session", "inverse_frequency"),
        )
        for seed in (42, 0, 1, 2, 3)
    }
    observed = set(
        matrix[["model", "experiment", "protocol", "weighting", "seed"]]
        .itertuples(index=False, name=None)
    )
    assert observed == expected


def test_committed_results_reproduce_headline_gap():
    xgb = json.loads(
        (ROOT / "results" / "xgboost" / "metrics" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    lgb = json.loads(
        (ROOT / "results" / "lightgbm" / "metrics" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert xgb["n_matched_classes"] == 15
    assert lgb["n_matched_classes"] == 15
    assert abs(xgb["gap_standard_mean"] - 0.589) < 0.002
    assert abs(xgb["gap_weighted_mean"] - 0.594) < 0.002
    assert abs(lgb["gap_standard_mean"] - 0.569) < 0.002
    assert abs(lgb["gap_weighted_mean"] - 0.571) < 0.002
    assert xgb["positive_gap_standard"] == 5
    assert lgb["positive_gap_weighted"] == 5
