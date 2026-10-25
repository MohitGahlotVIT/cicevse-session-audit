"""Recompute metrics directly from committed prediction files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = {"E1": ("random", "standard"), "E2": ("random", "inverse_frequency"), "E3": ("session", "standard"), "E4": ("session", "inverse_frequency")}


def _run_model(model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = ROOT / "results" / model
    metrics_dir = results_dir / "metrics"
    saved = pd.read_csv(metrics_dir / "run_results.csv")
    rows = []
    per_class_rows = []
    for path in sorted(
        path
        for path in (results_dir / "predictions").glob("E*_seed*.csv")
        if "_probe" not in path.name
    ):
        parts = path.stem.split("_")
        experiment = parts[0]
        seed = int(parts[1].replace("seed", ""))
        frame = pd.read_csv(path)
        labels = sorted(frame["y_true"].unique().tolist())
        result = compute_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy(), labels)
        protocol, weighting = EXPERIMENTS[experiment]
        rows.append(
            {
                "experiment": experiment,
                "protocol": protocol,
                "weighting": weighting,
                "seed": seed,
                "macro_f1": result["macro_f1"],
                "balanced_accuracy": result["balanced_accuracy"],
                "accuracy": result["accuracy"],
                "n_test": len(frame),
            }
        )
        for label, values in result["per_class"].items():
            per_class_rows.append(
                {
                    "experiment": experiment,
                    "protocol": protocol,
                    "seed": seed,
                    "class": label,
                    **values,
                }
            )
    recomputed = pd.DataFrame(rows)
    per_class = pd.DataFrame(per_class_rows)
    recomputed.to_csv(metrics_dir / "recomputed_run_metrics.csv", index=False)
    per_class.to_csv(metrics_dir / "recomputed_per_class_metrics.csv", index=False)

    merged = saved.merge(
        recomputed[["experiment", "seed", "macro_f1"]],
        on=["experiment", "seed"],
        suffixes=("_saved", "_recomputed"),
    )
    max_difference = float(
        np.abs(merged["macro_f1_saved"] - merged["macro_f1_recomputed"]).max()
    )
    assert max_difference < 1e-12, f"{model} prediction metrics do not match"
    print(f"{model}: {len(recomputed)} runs, max macro-F1 difference={max_difference:.2e}")
    return recomputed, per_class


def _matched(per_class: pd.DataFrame) -> pd.DataFrame:
    present = {
        experiment: set(group["class"])
        for experiment, group in per_class.groupby("experiment")
    }
    matched = sorted(set.intersection(*present.values()))
    filtered = per_class[per_class["class"].isin(matched)]
    run = (
        filtered.groupby(["experiment", "protocol", "seed"])["f1"]
        .mean()
        .reset_index()
        .rename(columns={"f1": "matched_macro_f1"})
    )
    run["n_matched_classes"] = len(matched)
    return run


def main() -> None:
    for model in ("xgboost", "lightgbm"):
        _, per_class = _run_model(model)
        matched = _matched(per_class)
        matched.to_csv(
            ROOT / "results" / model / "metrics" / "recomputed_matched_run_metrics.csv",
            index=False,
        )
        wide = matched.pivot(index="seed", columns="experiment", values="matched_macro_f1")
        wide["gap_standard"] = wide["E1"] - wide["E3"]
        wide["gap_weighted"] = wide["E2"] - wide["E4"]
        print(
            f"{model}: standard gap={wide['gap_standard'].mean():.3f} "
            f"+/- {wide['gap_standard'].std(ddof=1):.3f}; "
            f"weighted gap={wide['gap_weighted'].mean():.3f} "
            f"+/- {wide['gap_weighted'].std(ddof=1):.3f}"
        )


if __name__ == "__main__":
    main()
