"""Aggregate XGBoost run outputs and quantify the session-induced gap."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "results" / "xgboost" / "metrics"


def _matched_runs(per_class: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    present = {
        experiment: set(group["class"])
        for experiment, group in per_class.groupby("experiment")
    }
    matched_classes = sorted(set.intersection(*present.values()))
    matched = per_class[per_class["class"].isin(matched_classes)]
    run = (
        matched.groupby(["experiment", "protocol", "weighting", "seed"])[
            ["f1", "precision", "recall", "support"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "f1": "matched_macro_f1",
                "precision": "matched_macro_precision",
                "recall": "matched_macro_recall",
                "support": "mean_class_support",
            }
        )
    )
    run["n_matched_classes"] = len(matched_classes)
    wide = (
        matched.groupby(["class", "experiment"])["f1"]
        .mean()
        .unstack("experiment")
        .reset_index()
        .rename(columns={"class": "class"})
    )
    return run, wide


def _gap_summary(matched_run: pd.DataFrame) -> pd.DataFrame:
    wide = matched_run.pivot(
        index="seed", columns="experiment", values="matched_macro_f1"
    )
    rows = []
    for seed, values in wide.iterrows():
        rows.append(
            {
                "seed": int(seed),
                "E1_random_standard": values["E1"],
                "E2_random_weighted": values["E2"],
                "E3_session_standard": values["E3"],
                "E4_session_weighted": values["E4"],
                "gap_standard": values["E1"] - values["E3"],
                "gap_weighted": values["E2"] - values["E4"],
                "weighting_effect_random": values["E2"] - values["E1"],
                "weighting_effect_session": values["E4"] - values["E3"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    run = pd.read_csv(METRICS / "run_results.csv")
    per_class = pd.read_csv(METRICS / "per_class_metrics.csv")
    matched_run, matched_class = _matched_runs(per_class)
    gaps = _gap_summary(matched_run)

    matched_run.to_csv(METRICS / "matched_run_metrics.csv", index=False)
    matched_class.to_csv(METRICS / "matched_class_comparison.csv", index=False)
    gaps.to_csv(METRICS / "session_gap.csv", index=False)

    overall = (
        run.groupby(["protocol", "weighting"], sort=False)["macro_f1"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"count": "n_runs"})
    )
    overall.to_csv(METRICS / "aggregate_results.csv", index=False)

    diagnostic = json.loads(
        (METRICS / "session_diagnostic.json").read_text(encoding="utf-8")
    )
    probe = pd.read_csv(ROOT / "results" / "xgboost" / "predictions" / "probe_predictions.csv")

    summary = {
        "matched_classes": sorted(matched_class["class"].tolist()),
        "n_matched_classes": int(len(matched_class)),
        "overall": overall.to_dict(orient="records"),
        "matched_runs": matched_run.to_dict(orient="records"),
        "session_gap": gaps.to_dict(orient="records"),
        "gap_standard_mean": float(gaps["gap_standard"].mean()),
        "gap_standard_std": float(gaps["gap_standard"].std(ddof=1)),
        "gap_weighted_mean": float(gaps["gap_weighted"].mean()),
        "gap_weighted_std": float(gaps["gap_weighted"].std(ddof=1)),
        "positive_gap_standard": int((gaps["gap_standard"] > 0).sum()),
        "positive_gap_weighted": int((gaps["gap_weighted"] > 0).sum()),
        "weighting_effect_random_mean": float(gaps["weighting_effect_random"].mean()),
        "weighting_effect_random_std": float(gaps["weighting_effect_random"].std(ddof=1)),
        "weighting_effect_session_mean": float(gaps["weighting_effect_session"].mean()),
        "weighting_effect_session_std": float(gaps["weighting_effect_session"].std(ddof=1)),
        "diagnostic": diagnostic,
        "probe": probe.to_dict(orient="records"),
    }
    (METRICS / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"matched classes={len(matched_class)} "
        f"standard gap={summary['gap_standard_mean']:.3f} "
        f"+/- {summary['gap_standard_std']:.3f}; "
        f"weighted gap={summary['gap_weighted_mean']:.3f} "
        f"+/- {summary['gap_weighted_std']:.3f}"
    )


if __name__ == "__main__":
    main()
