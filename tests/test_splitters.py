import numpy as np

from src.splits.splitters import (
    class_session_table,
    make_random_split,
    make_session_split,
)


def _sessions(frame):
    return set(frame["session_id"].unique())


def _indices(splits):
    return {key: set(value.index) for key, value in splits.items() if len(value)}


def test_random_split_is_stratified_and_disjoint(synthetic_frame):
    splits = make_random_split(synthetic_frame, seed=42)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(synthetic_frame)
    train, val, test = (_indices(splits)[key] for key in ("train", "val", "test"))
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    for label in ("alpha", "beta", "gamma"):
        assert (splits["train"]["attack_label"] == label).any()
        assert (splits["test"]["attack_label"] == label).any()


def test_session_split_holds_out_whole_sessions(synthetic_frame):
    splits = make_session_split(synthetic_frame, seed=42)
    train_sessions = _sessions(splits["train"])
    val_sessions = _sessions(splits["val"])
    test_sessions = _sessions(splits["test"])
    assert not (train_sessions & test_sessions) - {0}
    assert not (val_sessions & test_sessions) - {0}
    assert 0 in train_sessions and 0 in val_sessions and 0 in test_sessions


def test_two_session_classes_hold_out_one_session(synthetic_frame):
    splits = make_session_split(synthetic_frame, seed=42)
    alpha_train = set(splits["train"].loc[splits["train"]["attack_label"] == "alpha", "session_id"])
    alpha_test = set(splits["test"].loc[splits["test"]["attack_label"] == "alpha", "session_id"])
    assert alpha_train == {1}
    assert alpha_test == {2}
    assert set(splits["val"].loc[splits["val"]["attack_label"] == "alpha", "session_id"]) == {1}


def test_three_session_classes_rotate(synthetic_frame):
    for seed, test_session in ((0, 5), (1, 6), (2, 7)):
        splits = make_session_split(synthetic_frame, seed=seed)
        gamma_test = set(
            splits["test"].loc[splits["test"]["attack_label"] == "gamma", "session_id"]
        )
        assert gamma_test == {test_session}


def test_probe_classes_excluded_from_training(synthetic_frame):
    splits = make_session_split(synthetic_frame, seed=42)
    probe_labels = {"os-scan", "icmp-fragmentation_old"}
    assert set(splits["probe"]["attack_label"]) == probe_labels
    for part in ("train", "val", "test"):
        assert not probe_labels & set(splits[part]["attack_label"])
    table = class_session_table(synthetic_frame)
    assert table.loc[table["class"] == "os-scan", "probe_only"].item()
    assert table.loc[table["class"] == "gamma", "n_sessions"].item() == 3
    assert table.loc[table["class"] == "none", "session0_timeslice"].item()
