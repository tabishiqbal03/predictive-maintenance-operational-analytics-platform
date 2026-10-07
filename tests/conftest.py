import numpy as np
import pandas as pd
import pytest

from maintenance.data import COLUMNS


@pytest.fixture(autouse=True)
def isolated_mlops(monkeypatch, tmp_path):
    # New telemetry must never write to the developer's production database during tests.
    monkeypatch.setenv("MAINTENANCE_ROOT", str(tmp_path))


@pytest.fixture
def frame():
    rows = []
    for engine, length in [(1, 8), (2, 6)]:
        for cycle in range(1, length + 1):
            rows.append([engine, cycle] + [float(cycle + engine + j) for j in range(24)])
    return pd.DataFrame(np.array(rows), columns=COLUMNS).astype({"engine_id": int, "cycle": int})
