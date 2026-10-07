"""Local integration smoke checks against generated artifacts; not required for CI."""

import json
import os
import signal
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import app
from maintenance.data import load
from maintenance.model import load_model

ROOT = Path(__file__).resolve().parents[1]


def check_server(command, url):
    log = ROOT / "artifacts" / ("server_" + str(command[2]) + ".log")
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=output, stderr=output, start_new_session=os.name != "nt"
        )
        try:
            for _ in range(60):
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited; inspect {log}")
                try:
                    with urlopen(url, timeout=2) as response:
                        assert response.status == 200
                    return
                except OSError:
                    time.sleep(0.5)
            raise RuntimeError(f"Server readiness timeout; inspect {log}")
        finally:
            if os.name == "nt":
                # MLflow starts a child server; terminating only its CLI leaves the port bound.
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)


def main():
    result = {}
    bundle = load_model(ROOT / "artifacts/best_model.joblib")
    assert set(bundle.metadata["fit_engines"]).isdisjoint(bundle.metadata["validation_engines"])
    assert len(bundle.metadata["fit_engines"]) == 75
    assert len(bundle.metadata["validation_engines"]) == 25
    frame = load(ROOT / "data/raw/test_FD001.txt")
    predictions = pd.read_csv(ROOT / "artifacts/predictions.csv").set_index("engine_id")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        for engine, history in frame.groupby("engine_id"):
            payload = {"observations": history.tail(bundle.window).to_dict("records")}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, response.text
            actual = response.json()
            np.testing.assert_allclose(
                actual["predicted_rul"], predictions.loc[engine, "predicted_rul"], rtol=1e-12
            )
            assert actual["maintenance_status"] == predictions.loc[engine, "maintenance_status"]
    result["api_batch_parity_engines"] = len(predictions)
    dashboard = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=30).run()
    assert not dashboard.exception, dashboard.exception
    for page in ["Asset Detail", "Model Performance", "Fleet Overview"]:
        dashboard.sidebar.radio[0].set_value(page).run()
        assert not dashboard.exception, dashboard.exception
    result["dashboard_pages"] = 3
    with sqlite3.connect(ROOT / "artifacts/maintenance.sqlite") as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM current_predictions").fetchone()[0] == 100
        assert conn.execute("SELECT COUNT(*) FROM engine_cycles").fetchone()[0] == len(frame)
    result["sqlite_integrity"] = "ok"
    check_server(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8765"],
        "http://127.0.0.1:8765/health",
    )
    result["uvicorn_http_health"] = "passed"
    check_server(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/dashboard.py",
            "--server.headless=true",
            "--server.address=127.0.0.1",
            "--server.port=8766",
            "--browser.gatherUsageStats=false",
        ],
        "http://127.0.0.1:8766/_stcore/health",
    )
    result["streamlit_http_health"] = "passed"
    (ROOT / "artifacts/runtime_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
