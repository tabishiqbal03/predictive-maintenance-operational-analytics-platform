"""Execute labelled artificial monitoring and verify local interfaces without changing production."""

import json
from pathlib import Path
import shutil
import sys

from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import app
from maintenance.mlops.monitoring import monitor, simulate
from maintenance.mlops.registry import active
from maintenance.mlops.store import digest, rows, write_json
from maintenance.mlops.tracking import client
from scripts.verify_runtime import check_server

ROOT = Path(__file__).resolve().parents[1]


def main():
    production = active(ROOT)
    manifests = list((ROOT / "artifacts/mlops").glob("benchmark_manifest_*.json"))
    for manifest in manifests:
        source_id = manifest.stem.removeprefix("benchmark_manifest_")
        current_id = json.loads((ROOT / "artifacts/metadata.json").read_text())["run_id"]
        base = ROOT / "artifacts" if source_id == current_id else ROOT / "artifacts/runs" / source_id
        for name, expected in json.loads(manifest.read_text()).items():
            assert digest(base / name) == expected, f"Benchmark snapshot changed: {name}"
    outcomes = []
    for scenario in ["normal", "sensor_bias", "operating_shift", "schema_failure"]:
        result = monitor(ROOT, simulate(ROOT, scenario))
        outcomes.append(result)
        print(scenario, result.get("data_drift"), result.get("performance_degraded"), flush=True)
    payload = json.loads((ROOT / "artifacts/example_request.json").read_text())
    with TestClient(app) as api:
        result = api.post("/predict", json=payload)
        assert result.status_code == 200
        assert result.json()["model_version"] == production["version"]
        assert api.post("/predict", json={"observations": []}).status_code == 422
        assert api.get("/model-info").json()["active_model_version"] == production["version"]
    dashboard = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=45).run()
    assert not dashboard.exception
    for page in ["Asset Detail", "Model Performance", "MLOps / Monitoring"]:
        dashboard.sidebar.radio[0].set_value(page).run()
        assert not dashboard.exception, dashboard.exception
    backend = "sqlite:///" + (ROOT / "artifacts/mlops/mlflow.sqlite").as_posix()
    check_server(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            backend,
            "--host",
            "127.0.0.1",
            "--port",
            "8767",
            "--workers",
            "1",
        ],
        "http://127.0.0.1:8767/health",
    )
    experiment = client(ROOT).get_experiment_by_name("FD001-RUL")
    run_count = len(client(ROOT).search_runs([experiment.experiment_id]))
    result = {
        "benchmark_hashes_unchanged": True,
        "active_version": production["version"],
        "mlflow_runs": run_count,
        "mlflow_server_http": "passed",
        "dashboard_pages": 4,
        "versioned_api_and_invalid_request": "passed",
        "scenarios": outcomes,
        "service_summary": rows(ROOT, "SELECT * FROM service_summary"),
        "docker": "CLI available; separate build required"
        if shutil.which("docker")
        else "not installed; build not verified",
    }
    write_json(ROOT / "artifacts/mlops/verification.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["scenarios", "service_summary"]}, indent=2)
    )


if __name__ == "__main__":
    main()
