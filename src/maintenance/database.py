"""Append-only run history with a current-fleet SQL view."""

import json
import sqlite3
from pathlib import Path

import pandas as pd


def populate(path: Path, raw: pd.DataFrame, predictions: pd.DataFrame, metadata: dict) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS model_runs (
            run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, metadata TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS model_predictions (
            run_id TEXT NOT NULL REFERENCES model_runs(run_id), engine_id INTEGER NOT NULL,
            cycle INTEGER NOT NULL, predicted_rul REAL NOT NULL CHECK(predicted_rul >= 0),
            actual_rul REAL, error REAL, maintenance_status TEXT NOT NULL, prediction_timestamp TEXT NOT NULL,
            PRIMARY KEY(run_id, engine_id, cycle));
        CREATE VIEW IF NOT EXISTS current_predictions AS
            SELECT * FROM model_predictions WHERE run_id =
            (SELECT run_id FROM model_runs ORDER BY created_at DESC LIMIT 1);
        CREATE VIEW IF NOT EXISTS maintenance_alerts AS SELECT * FROM current_predictions
            WHERE maintenance_status IN ('Critical', 'Schedule Maintenance');
        CREATE VIEW IF NOT EXISTS fleet_status AS SELECT maintenance_status, COUNT(*) AS assets,
            AVG(predicted_rul) AS mean_rul FROM current_predictions GROUP BY maintenance_status;
        """)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO model_runs VALUES (?,?,?)",
            (metadata["run_id"], metadata["created_at"], json.dumps(metadata)),
        )
        predictions.to_sql("model_predictions", conn, if_exists="append", index=False)
        raw.to_sql("engine_cycles", conn, if_exists="replace", index=False)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS cycle_key ON engine_cycles(engine_id, cycle)")
