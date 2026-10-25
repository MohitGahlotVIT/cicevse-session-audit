"""Compare LightGBM behaviour with the primary XGBoost results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.analyze_xgboost import _gap_summary, _matched_runs

ROOT = Path(__file__).resolve().parents[2]
XGB_METRICS = ROOT / "results" / "xgboost" / "metrics"
LGB_METRICS = ROOT / "results" / "lightgbm" / "metrics"


def _gaps_from_summary(path: Path) -> pd.DataFrame:
    summary = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(summary["session_gap"])


def main() -> None:
    lgb_run = pd.read_csv(LGB_METRICS / "run_results.csv")
    lgb_per_class = pd.read_csv(LGB_METRICS / "per_class_metrics.csv")
    matched_run, matched_class = _matched_runs(lgb_per_class)
    gaps = _gap_summary(matched_run)
    matched_run.to_csv(LGB_METRICS / "matched_run_metrics.csv", index=False)
    matched_class.to_csv(LGB_METRICS / "matched_class_comparison.csv", index=False)
    gaps.to_csv(LGB_METRICS / "session_gap.csv", index=False)

    xgb_gaps = _gaps_from_summary(XGB_METRICS / "summary.json")
    lgb_gaps = gaps

    model_rows = []
    for model, frame in (("xgboost", xgb_gaps), ("lightgbm", lgb_gaps)):
        for experiment in ("E1", "E2", "E3", "E4"):
            column = {
                "E1": "E1_random_standard",
                "E2": "E2_random_weighted",
                "E3": "E3_session_standard",
                "E4": "E4_session_weighted",
            }[experiment]
            model_rows.append(
                {
                    "model": model,
                    "experiment": experiment,
                    "macro_f1_mean": float(frame[column].mean()),
                    "macro_f1_std": float(frame[column].std(ddof=1)),
                    "n_seeds": int(len(frame)),
                }
            )
    model_comparison = pd.DataFrame(model_rows)
    model_comparison.to_csv(LGB_METRICS / "model_family_comparison.csv", index=False)

    gap_rows = []
    for gap_name in ("gap_standard", "gap_weighted"):
        gap_rows.append(
            {
                "gap": gap_name,
                "xgboost_mean": float(xgb_gaps[gap_name].mean()),
                "xgboost_std": float(xgb_gaps[gap_name].std(ddof=1)),
                "lightgbm_mean": float(lgb_gaps[gap_name].mean()),
                "lightgbm_std": float(lgb_gaps[gap_name].std(ddof=1)),
            }
        )
    gap_comparison = pd.DataFrame(gap_rows)
    gap_comparison.to_csv(LGB_METRICS / "model_family_gap_comparison.csv", index=False)

    overall = (
        lgb_run.groupby(["protocol", "weighting"], sort=False)["macro_f1"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"count": "n_runs"})
    )
    overall.to_csv(LGB_METRICS / "aggregate_results.csv", index=False)

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
        "model_family_comparison": model_comparison.to_dict(orient="records"),
        "model_family_gap_comparison": gap_comparison.to_dict(orient="records"),
    }
    (LGB_METRICS / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(gap_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
