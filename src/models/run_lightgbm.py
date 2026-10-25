"""Run the four LightGBM experiments over five seeds."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight

from src.data_prep.data_prep import load_config, prepare
from src.evaluation.metrics import compute_metrics
from src.splits.splitters import (
    class_session_table,
    coverage_report,
    make_random_split,
    make_session_split,
)

ROOT = Path(__file__).resolve().parents[2]


def _fit_lightgbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    weighted: bool,
) -> tuple[LGBMClassifier, SimpleImputer, list[str]]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[feature_cols])
    x_val = imputer.transform(val[feature_cols])
    classes = sorted(train["attack_label"].unique().tolist())
    class_to_int = {label: index for index, label in enumerate(classes)}
    y_train = train["attack_label"].map(class_to_int).to_numpy()
    y_val = val["attack_label"].map(class_to_int).to_numpy()
    assert not (set(val["attack_label"]) - set(classes)), "validation labels absent from training"

    params = {
        **params,
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "multi_logloss",
        "deterministic": True,
        "force_col_wise": True,
    }
    n_estimators = params.pop("n_estimators")
    early_stopping = params.pop("early_stopping_rounds", 30)
    classifier = LGBMClassifier(n_estimators=n_estimators, **params)
    fit_kwargs = {
        "eval_X": x_val,
        "eval_y": y_val,
        "callbacks": [
            lgb.early_stopping(early_stopping, verbose=False),
            lgb.log_evaluation(0),
        ],
    }
    if weighted:
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)
    classifier.fit(x_train, y_train, **fit_kwargs)
    return classifier, imputer, classes


def _predict(
    classifier: LGBMClassifier,
    imputer: SimpleImputer,
    frame: pd.DataFrame,
    feature_cols: list[str],
    classes: list[str],
) -> np.ndarray:
    encoded = classifier.predict(imputer.transform(frame[feature_cols])).astype(int)
    return np.asarray([classes[int(value)] for value in encoded])


def _save_confusion(path: Path, result: dict) -> None:
    labels = result["confusion_labels"]
    pd.DataFrame(result["confusion_matrix"], index=labels, columns=labels).to_csv(path)


def main() -> None:
    cfg = load_config()
    prepared = prepare(config=cfg)
    frame = prepared.frame
    features = prepared.feature_cols
    metrics_dir = ROOT / "results" / "lightgbm" / "metrics"
    predictions_dir = ROOT / "results" / "lightgbm" / "predictions"
    confusion_dir = metrics_dir / "confusion_matrices"
    for directory in (metrics_dir, predictions_dir, confusion_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seeds = cfg["splits"]["seeds"]
    ratios = tuple(cfg["splits"]["ratios"])
    run_rows = []
    per_class_rows = []
    coverage_rows = []
    probe_rows = []

    for experiment in cfg["experiments"]:
        experiment_id = experiment["id"]
        protocol = experiment["protocol"]
        weighted = experiment["weighting"] == "inverse_frequency"
        for seed in seeds:
            if protocol == "random":
                splits = make_random_split(frame, seed=seed, ratios=ratios)
            else:
                splits = make_session_split(frame, seed=seed, ratios=ratios)
            train = splits["train"]
            val = splits["val"]
            test = splits["test"]
            params = dict(cfg["models"]["lightgbm"])
            params["random_state"] = seed
            params["verbosity"] = -1
            n_estimators = params["n_estimators"]
            classifier, imputer, classes = _fit_lightgbm(
                train, val, features, params, weighted
            )
            y_true = test["attack_label"].to_numpy()
            y_pred = _predict(classifier, imputer, test, features, classes)
            test_labels = sorted(test["attack_label"].unique().tolist())
            result = compute_metrics(y_true, y_pred, test_labels)
            best_ntrees = int(classifier.best_iteration_ or n_estimators)

            run_rows.append(
                {
                    "experiment": experiment_id,
                    "protocol": protocol,
                    "weighting": experiment["weighting"],
                    "seed": seed,
                    "macro_f1": result["macro_f1"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "accuracy": result["accuracy"],
                    "weighted_f1": result["weighted_f1"],
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                    "n_features": len(features),
                    "n_classes_test": result["n_classes_evaluated"],
                    "best_ntrees": best_ntrees,
                }
            )
            for label, values in result["per_class"].items():
                per_class_rows.append(
                    {
                        "experiment": experiment_id,
                        "protocol": protocol,
                        "weighting": experiment["weighting"],
                        "seed": seed,
                        "class": label,
                        **values,
                    }
                )
            pd.DataFrame(
                {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "session_id": test["session_id"].to_numpy(),
                }
            ).to_csv(
                predictions_dir / f"{experiment_id}_seed{seed}.csv",
                index=False,
            )
            _save_confusion(
                confusion_dir / f"{experiment_id}_seed{seed}.csv", result
            )
            coverage_rows.append(
                coverage_report(splits, protocol, seed).assign(experiment=experiment_id)
            )

            if "probe" in splits and len(splits["probe"]):
                probe = splits["probe"]
                probe_pred = _predict(classifier, imputer, probe, features, classes)
                for label, group in probe.assign(prediction=probe_pred).groupby("attack_label"):
                    probe_rows.append(
                        {
                            "experiment": experiment_id,
                            "seed": seed,
                            "class": label,
                            "n_rows": len(group),
                            "n_predicted_attack": int((group["prediction"] != "none").sum()),
                            "fraction_predicted_attack": float(
                                (group["prediction"] != "none").mean()
                            ),
                        }
                    )

    run_frame = pd.DataFrame(run_rows)
    per_class_frame = pd.DataFrame(per_class_rows)
    run_frame.to_csv(metrics_dir / "run_results.csv", index=False)
    per_class_frame.to_csv(metrics_dir / "per_class_metrics.csv", index=False)
    pd.concat(coverage_rows, ignore_index=True).to_csv(
        metrics_dir / "coverage.csv", index=False
    )
    pd.DataFrame(probe_rows).to_csv(
        predictions_dir / "probe_predictions.csv", index=False
    )
    class_session_table(frame).to_csv(
        metrics_dir / "class_session_coverage.csv", index=False
    )

    aggregate = (
        run_frame.groupby(["protocol", "weighting"], sort=False)["macro_f1"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"count": "n_runs"})
    )
    aggregate.to_csv(metrics_dir / "aggregate_results.csv", index=False)

    manifest = {
        "model": "lightgbm",
        "prepared_shape": list(frame.shape),
        "n_features": len(features),
        "n_sessions": int(frame["session_id"].nunique()),
        "n_classes": int(frame["attack_label"].nunique()),
        "seeds": seeds,
        "experiments": cfg["experiments"],
        "prep_summary": prepared.log,
    }
    (metrics_dir / "prep_summary.json").write_text(
        json.dumps(prepared.log, indent=2), encoding="utf-8"
    )
    (metrics_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
