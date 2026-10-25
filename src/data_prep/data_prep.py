"""Deterministic preparation of the CICEVSE2024 host-event table.

The pipeline repairs the documented backdoor label displacement, normalises one
label typo, reconstructs capture sessions from timestamp resets, removes
non-informative columns, and returns the prepared modelling frame together with
the session assignment table and label-repair evidence.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiment_config.yaml"
META_COLS = ["time", "State", "Attack", "Scenario", "Label", "interface"]
REPAIR_COLUMNS = {
    "State": "interface",
    "Attack": "Unnamed: 911",
    "Scenario": "Unnamed: 912",
    "Label": "Unnamed: 913",
    "interface": "Unnamed: 914",
}


@dataclass(frozen=True)
class PreparedData:
    frame: pd.DataFrame
    feature_cols: list[str]
    log: dict
    session_assignments: pd.DataFrame
    label_repair_evidence: pd.DataFrame


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else CONFIG_PATH
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair_backdoor_shift(
    frame: pd.DataFrame,
    expected_rows: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Repair rows where the backdoor fields were shifted one column right."""
    df = frame.copy()
    mask = df["Attack"].astype(str) == "0"
    n_shifted = int(mask.sum())
    if expected_rows is not None:
        assert n_shifted == expected_rows, (
            f"expected {expected_rows} shifted rows, found {n_shifted}"
        )
    if n_shifted:
        for target, source in REPAIR_COLUMNS.items():
            df.loc[mask, target] = df.loc[mask, source].astype(str).values
    df["_label_repaired"] = mask
    return df, n_shifted


def assign_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign a session id whenever the timestamp moves backwards."""
    df = frame.copy()
    time_values = pd.to_numeric(df["time"], errors="coerce")
    valid = time_values.notna().to_numpy()
    ids = np.zeros(len(df), dtype=int)
    if valid.any():
        times = time_values.to_numpy()[valid]
        resets = np.where(np.diff(times) < 0)[0] + 1
        starts = np.concatenate(([0], resets))
        ends = np.concatenate((resets, [len(times)]))
        valid_ids = np.empty(len(times), dtype=int)
        for session_id, (start, end) in enumerate(zip(starts, ends)):
            valid_ids[start:end] = session_id
        ids[valid] = valid_ids
        last = 0
        for index in range(len(ids)):
            if valid[index]:
                last = ids[index]
            else:
                ids[index] = last
    df["session_id"] = ids
    return df


def select_feature_columns(columns: list[str]) -> list[str]:
    """Return model features while excluding metadata and internal columns."""
    return [
        column
        for column in columns
        if column not in META_COLS
        and column != "session_id"
        and not column.startswith("_")
    ]


def _session_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session_id, group in df.groupby("session_id", sort=True):
        classes = sorted(group["attack_label"].unique().tolist())
        scenarios = sorted(group["Scenario"].astype(str).unique().tolist())
        states = sorted(group["State"].astype(str).unique().tolist())
        times = pd.to_numeric(group["time"], errors="coerce")
        has_time = bool(times.notna().any())
        rows.append(
            {
                "session_id": int(session_id),
                "n_rows": int(len(group)),
                "n_classes": len(classes),
                "classes": ";".join(classes),
                "primary_class": group["attack_label"].mode().iloc[0],
                "scenarios": ";".join(scenarios),
                "states": ";".join(states),
                "time_start": int(times.min()) if has_time else "",
                "time_end": int(times.max()) if has_time else "",
                "duration": int(times.max() - times.min()) if has_time else "",
                "first_source_row": int(group.index[0]) + 2,
                "last_source_row": int(group.index[-1]) + 2,
            }
        )
    return pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)


def _repair_evidence(df: pd.DataFrame) -> pd.DataFrame:
    repaired = df[df["_label_repaired"]].copy()
    if repaired.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "state",
                "scenario",
                "repaired_class",
                "repaired_label",
                "repaired_interface",
                "n_rows",
                "first_source_row",
                "last_source_row",
            ]
        )
    rows = []
    group_cols = ["session_id", "State", "Scenario", "attack_label", "Label", "interface"]
    for keys, group in repaired.groupby(group_cols, sort=True):
        session_id, state, scenario, label, repaired_label, interface = keys
        rows.append(
            {
                "session_id": int(session_id),
                "state": state,
                "scenario": scenario,
                "repaired_class": label,
                "repaired_label": repaired_label,
                "repaired_interface": interface,
                "n_rows": int(len(group)),
                "first_source_row": int(group.index[0]) + 2,
                "last_source_row": int(group.index[-1]) + 2,
            }
        )
    return pd.DataFrame(rows)


def prepare(csv_path: str | Path | None = None, config: dict | None = None) -> PreparedData:
    """Prepare the host-event data and return the modelling frame and metadata."""
    cfg = config or load_config()
    dataset_cfg = cfg["dataset"]
    prep_cfg = cfg["preparation"]
    source = Path(csv_path or dataset_cfg["csv_path"])
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found at {source}. See data/README.md for download instructions."
        )

    log: dict = {
        "source_file": source.name,
        "sha256": sha256_of(source),
        "steps": [],
    }
    raw = pd.read_csv(source, low_memory=False)
    log["raw_shape"] = list(raw.shape)
    assert raw.shape == (dataset_cfg["expected_rows"], dataset_cfg["expected_cols"]), (
        f"unexpected raw shape: {raw.shape}"
    )
    assert log["sha256"] == dataset_cfg["expected_sha256"], "dataset checksum mismatch"

    df = raw.copy()
    blank = df["Attack"].isna()
    log["steps"].append({"drop_blank_attack_rows": int(blank.sum())})
    df = df[~blank].copy()

    garbage = df["Attack"].astype(str).str.startswith("writeback:")
    log["steps"].append({"drop_writeback_rows": int(garbage.sum())})
    df = df[~garbage].copy()

    df, n_shifted = repair_backdoor_shift(
        df, expected_rows=dataset_cfg["expected_repaired_rows"]
    )
    log["steps"].append({"repair_backdoor_shift_rows": n_shifted})

    if prep_cfg["merge_serice_into_service"]:
        typo = df["Attack"] == "serice-detection"
        log["steps"].append({"merge_serice_into_service": int(typo.sum())})
        df.loc[typo, "Attack"] = "service-detection"

    if prep_cfg["drop_exact_duplicates_keep_first"]:
        duplicates = int(df.duplicated().sum())
        log["steps"].append({"drop_exact_duplicates": duplicates})
        df = df.drop_duplicates(keep="first").copy()

    df = assign_sessions(df)
    log["steps"].append({"sessions_detected": int(df["session_id"].nunique())})

    drop_columns = [column for column in prep_cfg["drop_columns"] if column in df.columns]
    df = df.drop(columns=drop_columns)
    log["steps"].append({"drop_columns": drop_columns})

    feature_cols = select_feature_columns(df.columns.tolist())
    for column in feature_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    log["n_features_initial"] = len(feature_cols)

    if prep_cfg["drop_globally_constant_columns"]:
        nunique = df[feature_cols].nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        df = df.drop(columns=constant_cols)
        feature_cols = [column for column in feature_cols if column not in constant_cols]
        log["steps"].append({"drop_globally_constant_columns": len(constant_cols)})

    missing_fraction = df[feature_cols].isna().mean(axis=1)
    high_missing = missing_fraction > prep_cfg["max_missing_fraction_row"]
    log["steps"].append({"drop_high_missing_rows": int(high_missing.sum())})
    df = df[~high_missing].copy()

    df = df.rename(columns={"Attack": "attack_label"})
    session_assignments = _session_table(df)
    label_repair_evidence = _repair_evidence(df)

    prepared = df[["session_id", "attack_label"] + feature_cols].reset_index(drop=True)
    log["n_feature_cols"] = len(feature_cols)
    log["final_shape"] = [int(prepared.shape[0]), int(prepared.shape[1])]
    log["n_sessions"] = int(prepared["session_id"].nunique())
    log["n_classes"] = int(prepared["attack_label"].nunique())
    log["class_counts"] = {
        str(key): int(value)
        for key, value in prepared["attack_label"].value_counts().sort_index().items()
    }

    assert len(prepared) == dataset_cfg["prepared_rows"]
    assert len(feature_cols) == dataset_cfg["prepared_features"]
    assert prepared["attack_label"].nunique() == dataset_cfg["prepared_classes"]
    assert prepared["session_id"].nunique() == dataset_cfg["sessions"]

    return PreparedData(
        frame=prepared,
        feature_cols=feature_cols,
        log=log,
        session_assignments=session_assignments,
        label_repair_evidence=label_repair_evidence,
    )


def main() -> None:
    prepared = prepare()
    prepared.session_assignments.to_csv(
        ROOT / "data" / "session_assignments.csv", index=False
    )
    prepared.label_repair_evidence.to_csv(
        ROOT / "data" / "label_repair_evidence.csv", index=False
    )
    print(
        f"prepared={prepared.frame.shape[0]} rows, "
        f"{prepared.frame['session_id'].nunique()} sessions, "
        f"{prepared.frame['attack_label'].nunique()} classes, "
        f"{len(prepared.feature_cols)} features"
    )


if __name__ == "__main__":
    main()
