"""Regenerate the five result figures from committed metrics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "results" / "xgboost" / "metrics"
FIGURES = ROOT / "results" / "figures"


def _save(figure, name: str) -> None:
    figure.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def figure_gap() -> None:
    frame = pd.read_csv(METRICS / "matched_run_metrics.csv")
    order = ["E1", "E2", "E3", "E4"]
    labels = ["Random\nstandard", "Random\nweighted", "Session-aware\nstandard", "Session-aware\nweighted"]
    means = [frame.loc[frame["experiment"] == exp, "matched_macro_f1"].mean() for exp in order]
    stds = [frame.loc[frame["experiment"] == exp, "matched_macro_f1"].std(ddof=1) for exp in order]
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    colors = ["#3b6ea8", "#4a8b57", "#c0504d", "#d17c2a"]
    axis.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.6)
    for index, value in enumerate(means):
        axis.text(index, value + 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    axis.set_ylabel("Macro-F1 (15 matched classes)")
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    axis.set_axisbelow(True)
    figure.tight_layout()
    _save(figure, "fig1_matched_macro_f1_corrected_title.png")


def figure_recall() -> None:
    per_class = pd.read_csv(METRICS / "per_class_metrics.csv")
    matched = pd.read_csv(METRICS / "matched_class_comparison.csv")["class"].tolist()
    filtered = per_class[per_class["class"].isin(matched)]
    mean_recall = (
        filtered.groupby(["class", "experiment"])["recall"].mean().unstack("experiment")
    )
    mean_recall = mean_recall.sort_values("E1", ascending=True)
    positions = np.arange(len(mean_recall))
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.barh(positions - 0.19, mean_recall["E1"], height=0.38, color="#3b6ea8", label="Random split (E1)")
    axis.barh(positions + 0.19, mean_recall["E3"], height=0.38, color="#c0504d", label="Session-disjoint (E3)")
    axis.set_yticks(positions)
    axis.set_yticklabels(mean_recall.index, fontsize=8)
    axis.set_xlabel("Recall (mean over 5 seeds)")
    axis.set_xlim(0, 1.0)
    axis.legend(frameon=True, loc="lower right")
    axis.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    axis.set_axisbelow(True)
    figure.tight_layout()
    _save(figure, "fig2_per_class_recall.png")


def figure_confusion(experiment: str, seed: int, name: str, title: str) -> None:
    matrix = pd.read_csv(
        METRICS / "confusion_matrices" / f"{experiment}_seed{seed}.csv",
        index_col=0,
    )
    labels = matrix.index.tolist()
    values = matrix.to_numpy()
    figure, axis = plt.subplots(figsize=(7.4, 6.2))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title, fontsize=10)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Count")
    figure.tight_layout()
    _save(figure, name)


def figure_diagnostic() -> None:
    diagnostic = json.loads(
        (METRICS / "session_diagnostic.json").read_text(encoding="utf-8")
    )[0]
    labels = ["Session-ID\nprediction", "Majority\nbaseline"]
    values = [
        diagnostic["session_pred_accuracy"],
        diagnostic["majority_session_baseline"],
    ]
    figure, axis = plt.subplots(figsize=(5.6, 4.0))
    axis.bar(labels, values, color=["#6b5b95", "#9ca3af"], edgecolor="black", linewidth=0.6)
    for index, value in enumerate(values):
        axis.text(index, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    axis.set_axisbelow(True)
    figure.tight_layout()
    _save(figure, "fig5_session_diagnostic.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_gap()
    figure_recall()
    figure_confusion("E1", 42, "fig3_cm_E1_seed42.png", "Random split, standard XGBoost (E1)")
    figure_confusion("E3", 42, "fig4_cm_E3_seed42.png", "Session-disjoint, standard XGBoost (E3)")
    figure_diagnostic()
    print(f"wrote 5 figures to {FIGURES}")


if __name__ == "__main__":
    main()
