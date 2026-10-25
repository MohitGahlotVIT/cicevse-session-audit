import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    rows = []
    index = 0
    layout = [
        (0, "none", 40),
        (0, "cryptojacking", 20),
        (1, "alpha", 30),
        (2, "alpha", 15),
        (3, "beta", 30),
        (4, "beta", 15),
        (5, "gamma", 15),
        (6, "gamma", 15),
        (7, "gamma", 15),
        (8, "os-scan", 8),
        (9, "icmp-fragmentation_old", 8),
    ]
    for session_id, label, count in layout:
        for _ in range(count):
            rows.append(
                {
                    "session_id": session_id,
                    "attack_label": label,
                    "f1": float(index % 7),
                    "f2": float((index * 3) % 11),
                    "f3": float((index * 5) % 13),
                }
            )
            index += 1
    return pd.DataFrame(rows)
