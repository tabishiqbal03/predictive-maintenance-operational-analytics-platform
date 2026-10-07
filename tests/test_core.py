import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from maintenance.data import targets, validate
from maintenance.decision import status
from maintenance.evaluation import metrics, operational
from maintenance.features import engineer, informative
from maintenance.model import ModelBundle, load_model


def test_targets_and_validation(frame):
    clean = validate(frame.sample(frac=1, random_state=1))
    assert targets(clean).tolist() == list(range(7, -1, -1)) + list(range(5, -1, -1))


@pytest.mark.parametrize("issue", ["duplicate", "nan", "fractional", "gap", "schema", "non_numeric", "start"])
def test_bad_inputs(frame, issue):
    if issue == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif issue == "nan":
        frame.loc[0, "sensor_1"] = np.nan
    elif issue == "fractional":
        frame = frame.astype({"cycle": float})
        frame.loc[0, "cycle"] = 1.5
    elif issue == "gap":
        frame = frame.drop(2)
    elif issue == "schema":
        frame = frame.drop(columns="sensor_1")
    elif issue == "non_numeric":
        frame["sensor_1"] = "bad"
    else:
        frame = frame.drop(0)
    with pytest.raises(ValueError):
        validate(frame)


def test_features_are_causal_and_engine_local(frame):
    features = engineer(frame, ["sensor_1"], 3)
    prefix = frame[(frame.engine_id == 1) & (frame.cycle <= 4)]
    pd.testing.assert_frame_equal(features.loc[prefix.index], engineer(prefix, ["sensor_1"], 3))
    changed = frame.copy()
    changed.loc[(changed.engine_id == 2) | (changed.cycle > 4), "sensor_1"] = 9999
    pd.testing.assert_frame_equal(
        features.loc[prefix.index], engineer(changed, ["sensor_1"], 3).loc[prefix.index]
    )
    assert features.loc[8, "sensor_1_std"] == 0
    assert features.loc[8, "sensor_1_delta"] == 0
    assert features.loc[2, "sensor_1_mean"] == frame.loc[:2, "sensor_1"].mean()
    assert features.loc[3, "sensor_1_delta"] == 1


def test_channel_selection_training_only(frame):
    fit = frame[frame.engine_id == 1].copy()
    fit["sensor_1"] = 10
    assert "sensor_1" not in informative(fit)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "Critical"),
        (20, "Critical"),
        (20.1, "Schedule Maintenance"),
        (40, "Schedule Maintenance"),
        (40.1, "Monitor"),
        (80, "Monitor"),
        (80.1, "Healthy"),
    ],
)
def test_status(value, expected):
    assert status(value) == expected


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_invalid_rul(value):
    with pytest.raises(ValueError):
        status(value)


def test_score_and_operational_definitions():
    result = metrics([20, 20], [7, 30])
    assert result["nasa_score"] == pytest.approx(2 * np.expm1(1))
    assert metrics([20], [30])["nasa_score"] > metrics([20], [10])["nasa_score"]
    result = operational([10, 15, 80], [10, 50, 30])
    assert result["missed_critical"] == 1
    assert result["false_alerts"] == 1
    assert result["near_failure_recall"] == 0.5


def test_model_roundtrip_and_serving_history(frame, tmp_path):
    x = engineer(frame, ["sensor_1"], 3)
    model = LinearRegression().fit(x, targets(frame))
    bundle = ModelBundle(model, ["sensor_1"], 3, {"run_id": "test"})
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    loaded = load_model(path)
    np.testing.assert_allclose(bundle.predict(frame), loaded.predict(frame))
    history = frame[frame.engine_id == 1]
    assert loaded.latest(history) == loaded.latest(history.tail(3))
    with pytest.raises(ValueError, match="at least"):
        loaded.latest(history.tail(1))
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing")
