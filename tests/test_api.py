import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from api.main import app, get_model
from maintenance.features import engineer
from maintenance.model import ModelBundle


@pytest.fixture
def client(frame):
    model = DummyRegressor(strategy="constant", constant=35).fit(
        engineer(frame, ["sensor_1"], 3), frame.cycle
    )
    app.dependency_overrides.clear()
    original = get_model
    import api.main as serving

    serving.get_model = lambda: ModelBundle(model, ["sensor_1"], 3, {"run_id": "fixture"})
    with TestClient(app) as client:
        yield client
    serving.get_model = original


def test_health_and_prediction(client, frame):
    assert client.get("/health").json()["status"] == "ready"
    assert client.get("/model-info").json()["required_recent_cycles"] == 3
    payload = {"observations": frame[frame.engine_id == 1].tail(3).to_dict("records")}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_rul"] == 35
    assert response.json()["maintenance_status"] == "Schedule Maintenance"


def test_bad_requests(client, frame):
    assert client.post("/predict", json={"observations": []}).status_code == 422
    assert client.post("/predict", json={"observations": [{"engine_id": 1}]}).status_code == 422
    assert client.post("/predict", json={"observations": frame.to_dict("records")}).status_code == 422
    rows = frame[frame.engine_id == 1].tail(3).to_dict("records")
    rows[-1]["cycle"] += 2
    assert client.post("/predict", json={"observations": rows}).status_code == 422
    rows[-1]["unexpected"] = 1
    assert client.post("/predict", json={"observations": rows}).status_code == 422


def test_unavailable_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MAINTENANCE_MODEL", str(tmp_path / "missing.joblib"))
    get_model.cache_clear()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
    get_model.cache_clear()
