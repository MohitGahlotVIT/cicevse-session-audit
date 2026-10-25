from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep.data_prep import assign_sessions, repair_backdoor_shift

ROOT = Path(__file__).resolve().parents[1]


def test_repair_shift_maps_fields():
    frame = pd.DataFrame(
        {
            "time": [1, 2],
            "State": ["Charging", "Charging"],
            "Attack": ["0", "backdoor"],
            "Scenario": ["Benign", "host-attack"],
            "Label": ["none", "attack"],
            "interface": ["Charging", "any"],
            "Unnamed: 911": ["host-attack", "host-attack"],
            "Unnamed: 912": ["attack", "attack"],
            "Unnamed: 913": ["any", "any"],
            "Unnamed: 914": ["any", "any"],
        }
    )
    repaired, count = repair_backdoor_shift(frame)
    assert count == 1
    assert repaired.loc[0, "State"] == "Charging"
    assert repaired.loc[0, "Attack"] == "host-attack"
    assert repaired.loc[0, "Scenario"] == "attack"
    assert repaired.loc[0, "Label"] == "any"
    assert repaired.loc[0, "interface"] == "any"
    assert repaired.loc[1, "Attack"] == "backdoor"
    assert repaired["_label_repaired"].tolist() == [True, False]


def test_assign_sessions_on_time_reset():
    frame = pd.DataFrame({"time": [5, 6, 5, 7, 8]})
    assigned = assign_sessions(frame)
    assert assigned["session_id"].tolist() == [0, 0, 1, 1, 1]


def test_session_assignments_cover_prepared_rows():
    assignments = pd.read_csv(ROOT / "data" / "session_assignments.csv")
    assert len(assignments) == 36
    assert int(assignments["n_rows"].sum()) == 8468
    assert assignments.loc[assignments["session_id"] == 0, "n_rows"].item() == 4095
    assert assignments.loc[assignments["session_id"] == 34, "n_rows"].item() == 1151
    assert assignments.loc[assignments["session_id"] == 35, "n_rows"].item() == 1151
    assert assignments["session_id"].is_unique


def test_label_repair_evidence_exact():
    evidence = pd.read_csv(ROOT / "data" / "label_repair_evidence.csv")
    assert len(evidence) == 2
    assert int(evidence["n_rows"].sum()) == 2302
    assert set(evidence["repaired_class"]) == {"backdoor"}
    assert set(evidence["state"]) == {"Charging", "idle"}
    assert set(evidence["scenario"]) == {"host-attack"}
    assert set(evidence["repaired_label"]) == {"attack"}
    assert set(evidence["repaired_interface"]) == {"any"}
    first = evidence[evidence["state"] == "Charging"].iloc[0]
    second = evidence[evidence["state"] == "idle"].iloc[0]
    assert (int(first["first_source_row"]), int(first["last_source_row"])) == (6173, 7323)
    assert (int(second["first_source_row"]), int(second["last_source_row"])) == (7324, 8475)
    assert np.isfinite(evidence["n_rows"]).all()
