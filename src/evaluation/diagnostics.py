"""Session-confounding diagnostic.

The diagnostic fits a model that predicts capture-session identity from host
features within the training partition. Most sessions contain a single attack
class, so the result is reported alongside a session-by-class contingency table
and is not interpreted as proof of feature leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def session_contingency(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(frame["session_id"], frame["attack_label"])


def session_identity_diagnostic(
    train: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 42,
) -> dict:
    session_counts = train["session_id"].value_counts()
    keep = session_counts[session_counts >= 20].index
    subset = train[train["session_id"].isin(keep)].copy()
    if subset["session_id"].nunique() < 2:
        return {"status": "skipped", "reason": "fewer than 2 sessions with >=20 rows"}

    imputer = SimpleImputer(strategy="median")
    features = imputer.fit_transform(subset[feature_cols])
    session_values = subset["session_id"].to_numpy()
    unique_sessions = sorted(np.unique(session_values).tolist())
    encoder = {value: index for index, value in enumerate(unique_sessions)}
    encoded = np.asarray([encoder[value] for value in session_values])

    x_train, x_test, y_train, y_test = train_test_split(
        features, encoded, test_size=0.20, random_state=seed, stratify=encoded
    )
    classifier = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.10,
        subsample=0.80,
        colsample_bytree=0.80,
        tree_method="hist",
        num_class=len(unique_sessions),
        random_state=seed,
        n_jobs=4,
        verbosity=0,
    )
    classifier.fit(x_train, y_train)
    accuracy = float(accuracy_score(y_test, classifier.predict(x_test)))
    baseline = float(pd.Series(y_test).value_counts(normalize=True).max())
    contingency = session_contingency(subset)
    single_class = float((contingency.astype(bool).sum(axis=1) == 1).mean())

    return {
        "status": "ok",
        "n_sessions_modelled": int(subset["session_id"].nunique()),
        "n_rows": int(len(subset)),
        "session_pred_accuracy": accuracy,
        "majority_session_baseline": baseline,
        "lift_over_baseline": accuracy - baseline,
        "frac_single_class_sessions": single_class,
        "caveat": (
            "session label is confounded with attack class "
            "(most sessions contain one class); accuracy cannot be interpreted "
            "as pure session-fingerprint leakage"
        ),
    }
