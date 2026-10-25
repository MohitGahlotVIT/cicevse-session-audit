"""Random-row and capture-session-disjoint split construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

PROBE_CLASSES = {"os-scan", "icmp-fragmentation_old"}
SESSION0_CLASSES = {"none", "cryptojacking"}


def class_session_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in df.groupby("attack_label"):
        sessions = sorted(group["session_id"].unique().tolist())
        rows.append(
            {
                "class": label,
                "n_rows": len(group),
                "n_sessions": len(sessions),
                "sessions": sessions,
                "eligible_known": len(sessions) >= 2 and label not in PROBE_CLASSES,
                "probe_only": label in PROBE_CLASSES,
                "session0_timeslice": label in SESSION0_CLASSES,
            }
        )
    return pd.DataFrame(rows).sort_values("n_sessions", ascending=False)


def make_random_split(
    df: pd.DataFrame,
    seed: int,
    ratios: tuple[float, float, float] = (0.60, 0.20, 0.20),
) -> dict[str, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(df))
    labels = df["attack_label"].to_numpy()
    train_idx, rest_idx = train_test_split(
        indices, train_size=ratios[0], random_state=seed, stratify=labels
    )
    validation_fraction = ratios[1] / (ratios[1] + ratios[2])
    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=validation_fraction,
        random_state=seed,
        stratify=labels[rest_idx],
    )
    return {
        "train": df.iloc[train_idx].copy(),
        "val": df.iloc[val_idx].copy(),
        "test": df.iloc[test_idx].copy(),
    }


def make_session_split(
    df: pd.DataFrame,
    seed: int,
    ratios: tuple[float, float, float] = (0.60, 0.20, 0.20),
) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    probe_parts: list[pd.DataFrame] = []

    for label, group in df.groupby("attack_label"):
        if label in PROBE_CLASSES:
            probe_parts.append(group)
            continue
        if label in SESSION0_CLASSES:
            continue

        sessions = sorted(group["session_id"].unique().tolist())
        assert len(sessions) >= 2, f"{label} has fewer than two sessions"

        if len(sessions) == 2:
            first, second = sessions
            train_session, test_session = (first, second) if seed % 2 == 0 else (second, first)
            train_group = group[group["session_id"] == train_session]
            n_val = max(
                1,
                int(round(len(train_group) * ratios[1] / (ratios[0] + ratios[1]))),
            )
            train_parts.append(train_group.iloc[:-n_val])
            val_parts.append(train_group.iloc[-n_val:])
            test_parts.append(group[group["session_id"] == test_session])
        else:
            rotation = seed % 3
            test_session = sessions[rotation]
            val_session = sessions[(rotation + 1) % 3]
            train_session = sessions[(rotation + 2) % 3]
            train_parts.append(group[group["session_id"] == train_session])
            val_parts.append(group[group["session_id"] == val_session])
            test_parts.append(group[group["session_id"] == test_session])

    session_zero = df[df["session_id"] == 0].sort_index()
    n_train = int(round(len(session_zero) * ratios[0]))
    n_val = int(round(len(session_zero) * ratios[1]))
    train_parts.append(session_zero.iloc[:n_train])
    val_parts.append(session_zero.iloc[n_train : n_train + n_val])
    test_parts.append(session_zero.iloc[n_train + n_val :])

    train = pd.concat(train_parts).sample(frac=1, random_state=rng.randint(1 << 31))
    val = pd.concat(val_parts)
    test = pd.concat(test_parts)
    probe = pd.concat(probe_parts) if probe_parts else df.iloc[0:0]

    def sessions(frame: pd.DataFrame) -> set[int]:
        return set(frame["session_id"].unique())

    train_sessions, val_sessions, test_sessions = (
        sessions(train),
        sessions(val),
        sessions(test),
    )
    assert not (train_sessions & test_sessions) - {0}
    assert not (val_sessions & test_sessions) - {0}

    train_rows, val_rows, test_rows = set(train.index), set(val.index), set(test.index)
    assert not (train_rows & val_rows)
    assert not (train_rows & test_rows)
    assert not (val_rows & test_rows)

    probe_sessions = sessions(probe)
    assert not (probe_sessions & train_sessions)
    assert not (set(probe.index) & (train_rows | val_rows | test_rows))

    return {"train": train, "val": val, "test": test, "probe": probe}


def coverage_report(
    splits: dict[str, pd.DataFrame], protocol: str, seed: int
) -> pd.DataFrame:
    rows = []
    labels = sorted(
        set(splits["test"]["attack_label"]) | set(splits["train"]["attack_label"])
    )
    for label in labels:
        row = {"protocol": protocol, "seed": seed, "class": label}
        for part in ("train", "val", "test", "probe"):
            if part in splits:
                row[f"n_{part}"] = int((splits[part]["attack_label"] == label).sum())
            else:
                row[f"n_{part}"] = 0
        row["in_test"] = row["n_test"] > 0
        row["in_train"] = row["n_train"] > 0
        rows.append(row)
    return pd.DataFrame(rows)
