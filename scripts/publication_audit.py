"""Read-only preservation snapshot for the targeted publication pass."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    paths = list((ROOT / "artifacts/mlops").rglob("*"))
    paths += [
        ROOT / "artifacts" / name
        for name in [
            "best_model.joblib",
            "metrics.json",
            "predictions.csv",
            "metadata.json",
            "model_comparison.csv",
        ]
    ]
    files = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.is_file() and p.suffix not in [".sqlite", ".log"]
    }
    with sqlite3.connect(ROOT / "artifacts/maintenance.sqlite") as db:
        state = {
            name: db.execute(f"SELECT * FROM {name}").fetchall()
            for name in [
                "production_model",
                "model_versions",
                "lifecycle_events",
                "challenger_evaluations",
                "test_attestations",
                "monitoring_runs",
            ]
        }
    with sqlite3.connect(ROOT / "artifacts/mlops/mlflow.sqlite") as db:
        tracking = {
            name: db.execute(f"SELECT * FROM {name}").fetchall()
            for name in ["experiments", "runs", "model_versions", "registered_models"]
        }
    return json.loads(json.dumps({"files": files, "state": state, "tracking": tracking}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["before", "after", "inspect", "smoke"])
    args = parser.parse_args()
    path = ROOT / "artifacts/publication_before.json"
    if args.mode == "inspect":
        from maintenance.mlops.tracking import client

        api = client(ROOT)
        experiment = api.get_experiment_by_name("FD001-RUL")
        runs = api.search_runs([experiment.experiment_id])
        print("Runs:", len(runs), "finished:", sum(r.info.status == "FINISHED" for r in runs))
        print("Historical artifact location:", experiment.artifact_location)
        print("Versions:", [(v.version, v.source) for v in api.search_model_versions("name='FD001-RUL'")])
        return
    if args.mode == "smoke":
        import sys
        from fastapi.testclient import TestClient
        from streamlit.testing.v1 import AppTest
        from api.main import app
        from scripts.verify_runtime import check_server

        with TestClient(app) as api:
            assert api.get("/health").json()["model_version"] == "1"
            assert api.get("/model-info").json()["active_model_version"] == "1"
        dashboard = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=45).run()
        assert not dashboard.exception
        for page in ["Asset Detail", "Model Performance", "MLOps / Monitoring"]:
            dashboard.sidebar.radio[0].set_value(page).run()
            assert not dashboard.exception, dashboard.exception
            if page == "Asset Detail":
                dashboard.checkbox[0].check().run()
                assert not dashboard.exception
        check_server(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--backend-store-uri",
                "sqlite:///" + (ROOT / "artifacts/mlops/mlflow.sqlite").as_posix(),
                "--host",
                "127.0.0.1",
                "--port",
                "8777",
                "--workers",
                "1",
            ],
            "http://127.0.0.1:8777/health",
        )
        print("Four dashboard areas, retrospective toggle, API version 1 and MLflow server readiness passed.")
        return
    current = snapshot()
    if args.mode == "before":
        if path.exists():
            raise ValueError("Preservation snapshot already exists; refusing to replace it")
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    else:
        assert json.loads(path.read_text()) == current, "Protected artifacts or lifecycle records changed"
        print("Protected model/benchmark/history files and registry/MLflow records unchanged.")


if __name__ == "__main__":
    main()
