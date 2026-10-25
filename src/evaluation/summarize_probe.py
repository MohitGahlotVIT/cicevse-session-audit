"""Summarise probe predictions for classes seen in only one session."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def summarize(model: str) -> pd.DataFrame:
    predictions = ROOT / "results" / model / "predictions"
    rows = []
    for path in sorted(predictions.glob("E*_seed*_probe.csv")):
        experiment, seed_text = path.stem.split("_")[:2]
        seed = int(seed_text.replace("seed", ""))
        frame = pd.read_csv(path)
        for label, group in frame.groupby("y_true"):
            rows.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "class": label,
                    "n_rows": len(group),
                    "n_predicted_attack": int((group["y_pred"] != "none").sum()),
                    "fraction_predicted_attack": float((group["y_pred"] != "none").mean()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(predictions / "probe_predictions.csv", index=False)
    return summary


def main() -> None:
    for model in ("xgboost", "lightgbm"):
        summary = summarize(model)
        print(f"{model}: {len(summary)} probe summaries")


if __name__ == "__main__":
    main()
