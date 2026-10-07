import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import ExtraTreesRegressor

from maintenance.data import targets
from maintenance.features import engineer
from maintenance.model import ModelBundle
from maintenance.mlops import registry
from maintenance.mlops.lifecycle import gates, retrain
from maintenance.mlops.monitoring import drift, monitor, simulate
from maintenance.mlops.store import connection, digest, initialize, now, rows, source_hash
from maintenance.mlops.tracking import client, log_run


@pytest.fixture
def registered(frame, tmp_path, monkeypatch):
    initialize(tmp_path)
    monkeypatch.setattr(
        registry,
        "register_mlflow",
        lambda root, run, path: str(len(rows(root, "SELECT * FROM model_versions")) + 1),
    )
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    frame.to_csv(raw / "train_FD001.txt", sep=" ", index=False, header=False)
    model = ExtraTreesRegressor(n_estimators=3, random_state=42).fit(
        engineer(frame, ["sensor_1"], 3), targets(frame)
    )
    meta = {
        "run_id": "source",
        "source_run_id": "source",
        "feature_version": "causal-rolling-v1",
        "candidate": "fixture",
        "imported_benchmark": True,
        "fit_engines": [1],
        "validation_engines": [2],
        "training_sha256": digest(raw / "train_FD001.txt"),
        "cap": None,
        "benchmark": {"mae": 1},
    }
    model_file = tmp_path / "model.joblib"
    joblib.dump(ModelBundle(model, ["sensor_1"], 3, meta), model_file)
    version = registry.register(tmp_path, model_file, "run-fixture", meta)
    registry.switch(tmp_path, version, "test initial approval", bootstrap=True)
    return tmp_path, model_file, meta


def candidate(registered):
    root, path, meta = registered
    version = registry.register(root, path, "challenger-run", {**meta, "imported_benchmark": False})
    return version


def evidence(root, version, eligible=True, champion="1"):
    record = registry.get_version(root, version)
    with connection(root) as db:
        db.execute(
            "INSERT OR REPLACE INTO challenger_evaluations VALUES (?,?,?,?,?)",
            (version, champion, now(), eligible, "{}"),
        )
        db.execute(
            "INSERT OR REPLACE INTO test_attestations VALUES (?,?,?,?,?,?)",
            (version, now(), source_hash(root), record["sha256"], 1, "test-fixture-only.xml"),
        )


def test_mlflow_logging_and_real_registration(tmp_path, frame):
    model = ExtraTreesRegressor(n_estimators=2).fit(engineer(frame, ["sensor_1"], 3), targets(frame))
    path = tmp_path / "model.joblib"
    joblib.dump(ModelBundle(model, ["sensor_1"], 3, {"run_id": "real-mlflow"}), path)
    run = log_run(tmp_path, "integration", {"window": 3}, {"rmse": 2.5}, [path])
    recorded = client(tmp_path).get_run(run)
    assert recorded.info.status == "FINISHED"
    assert recorded.data.metrics["rmse"] == 2.5
    assert client(tmp_path).list_artifacts(run, "evidence")[0].path.endswith("model.joblib")
    version = registry.register(tmp_path, path, run, {"feature_version": "fixture"})
    assert client(tmp_path).get_model_version("FD001-RUL", version).run_id == run


def test_registry_promotion_and_rollback(registered):
    root, _, _ = registered
    version = candidate(registered)
    assert registry.active(root)["version"] == "1"
    evidence(root, version)
    registry.switch(root, version, "qualified candidate")
    assert registry.active(root)["version"] == version
    assert registry.get_version(root, "1")["lifecycle"] == "previous"
    registry.switch(root, "1", "restore previous approval", rollback=True)
    assert registry.active(root)["version"] == "1"
    assert [r["action"] for r in rows(root, "SELECT * FROM lifecycle_events")] == [
        "bootstrap",
        "promotion",
        "rollback",
    ]


@pytest.mark.parametrize(
    "problem", ["no_tests", "failed_gates", "stale_champion", "changed_source", "corrupt_model"]
)
def test_promotion_rejects_bad_evidence(registered, problem):
    root, _, _ = registered
    version = candidate(registered)
    evidence(
        root,
        version,
        eligible=problem != "failed_gates",
        champion="999" if problem == "stale_champion" else "1",
    )
    if problem == "no_tests":
        with connection(root) as db:
            db.execute("DELETE FROM test_attestations")
    elif problem == "changed_source":
        (root / "mlops.py").write_text("changed")
    elif problem == "corrupt_model":
        record = registry.get_version(root, version)
        (root / record["model_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        registry.switch(root, version, "must fail")
    assert registry.active(root)["version"] == "1"


def test_unapproved_rollback_rejected(registered):
    root, _, _ = registered
    with pytest.raises(ValueError, match="never approved"):
        registry.switch(root, candidate(registered), "not allowed", rollback=True)


def test_gates_do_not_force_improvement():
    champion = {
        "mae": 20,
        "rmse": 30,
        "nasa_score": 100,
        "near_failure_recall": 0.95,
        "false_alerts": 2,
        "missed_critical": 1,
    }
    assert gates(champion, champion)["eligible"]
    for change in [{"rmse": 40}, {"near_failure_recall": 0.8}, {"false_alerts": 4}, {"missed_critical": 2}]:
        assert not gates(champion, {**champion, **change})["eligible"]


def test_effect_size_drift():
    reference = pd.DataFrame({"sensor": np.linspace(-1, 1, 100)})
    assert not drift(reference, reference).drifted.any()
    assert drift(reference, reference + 3).drifted.all()


@pytest.mark.parametrize("scenario", ["normal", "sensor_bias", "operating_shift", "schema_failure"])
def test_monitoring_reports(registered, scenario):
    root, _, _ = registered
    folder = simulate(root, scenario)
    result = monitor(root, folder)
    assert result["model_version"] == "1"
    report = rows(root, "SELECT * FROM monitoring_runs")[0]
    assert (root / report["report_path"] / "report.md").exists()
    if scenario == "schema_failure":
        assert result["status"] == "invalid"
        assert result["performance_degraded"] is None
    else:
        assert np.isfinite(result["performance"]["rmse"])
        assert rows(root, "SELECT COUNT(*) AS n FROM inference_events")[0]["n"] == 2
        if scenario == "sensor_bias":
            # Fixture estimator uses sensor_1 only: changes to unused channels are still drift,
            # but must not be mislabeled as a performance regression.
            assert result["data_drift"]
            assert not result["prediction_drift"]
            assert not result["performance_degraded"]


def test_unlabelled_monitor_does_not_claim_accuracy(registered):
    root, _, _ = registered
    result = monitor(root, simulate(root, "normal"), labelled=False)
    assert "performance" not in result
    assert result["performance_degraded"] is None


def test_versioned_api_switches_and_logs_invalid_requests(registered, frame):
    from api.main import app, get_model

    root, _, _ = registered
    get_model.cache_clear()
    with TestClient(app) as api:
        assert api.get("/model-info").json()["active_model_version"] == "1"
        payload = {"observations": frame[frame.engine_id == 1].tail(3).to_dict("records")}
        assert api.post("/predict", json=payload).json()["model_version"] == "1"
        version = candidate(registered)
        evidence(root, version)
        registry.switch(root, version, "reload test")
        assert api.post("/predict", json=payload).json()["model_version"] == version
        assert api.post("/predict", json={"observations": []}).status_code == 422
    events = rows(root, "SELECT * FROM inference_events ORDER BY id")
    assert [e["status_code"] for e in events] == [200, 200, 422]
    assert all(e["latency_ms"] >= 0 for e in events)
    assert events[-1]["model_version"] == version


def test_train_only_retraining(registered, monkeypatch):
    root, _, _ = registered
    import maintenance.mlops.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "log_run", lambda *args, **kwargs: "test-retrain")
    # No test_FD001 or RUL file exists in this fixture: retraining must still work.
    result = retrain(root)
    assert result["version"] == "2"
    assert np.isfinite(result["challenger"]["rmse"])
    assert registry.active(root)["version"] == "1"
    assert rows(root, "SELECT * FROM challenger_evaluations")[0]["version"] == "2"


def test_qualify_requires_successful_test_process(registered, monkeypatch):
    from types import SimpleNamespace
    from maintenance.mlops.lifecycle import qualify

    root, _, _ = registered
    monkeypatch.setattr(
        "maintenance.mlops.lifecycle.subprocess.run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout="failed", stderr=""),
    )
    with pytest.raises(ValueError, match="Tests failed"):
        qualify(root, "1")
    assert not rows(root, "SELECT * FROM test_attestations")
