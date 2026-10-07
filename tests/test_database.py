import sqlite3

import pandas as pd

from maintenance.database import populate


def test_latest_run_and_history(frame, tmp_path):
    path = tmp_path / "fleet.sqlite"
    for run, day in [("first", "2026-01-01"), ("second", "2026-01-02")]:
        predictions = pd.DataFrame(
            [
                dict(
                    run_id=run,
                    engine_id=1,
                    cycle=8,
                    predicted_rul=10,
                    actual_rul=12,
                    error=-2,
                    maintenance_status="Critical",
                    prediction_timestamp=day,
                )
            ]
        )
        populate(path, frame, predictions, {"run_id": run, "created_at": day})
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0] == 2
        assert conn.execute("SELECT run_id FROM maintenance_alerts").fetchone()[0] == "second"
        assert conn.execute("SELECT assets FROM fleet_status").fetchone()[0] == 1
