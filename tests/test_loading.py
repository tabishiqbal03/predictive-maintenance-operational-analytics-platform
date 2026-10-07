import io
import json
from zipfile import ZipFile

import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from maintenance.data import FILES, download, load, targets
from maintenance.features import engineer


def test_raw_load_roundtrip_and_missing(frame, tmp_path):
    path = tmp_path / "train_FD001.txt"
    frame.to_csv(path, sep=" ", header=False, index=False)
    pd.testing.assert_frame_equal(load(path), frame)
    with pytest.raises(FileNotFoundError, match="--download"):
        load(tmp_path / "missing")


def test_download_nested_archive(monkeypatch, frame, tmp_path):
    inner = io.BytesIO()
    with ZipFile(inner, "w") as archive:
        for name in FILES:
            archive.writestr("CMAPSSData/" + name, "fixture data")
    outer = io.BytesIO()
    with ZipFile(outer, "w") as archive:
        archive.writestr("CMAPSSData.zip", inner.getvalue())
    import maintenance.data as data

    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(outer.getvalue()))
    download(tmp_path)
    assert all((tmp_path / name).read_text() == "fixture data" for name in FILES)
    assert set(json.loads((tmp_path / "provenance.json").read_text())["sha256"]) == set(FILES)


def test_scaler_retains_fitting_statistics(frame):
    train = frame[frame.engine_id == 1]
    val = frame[frame.engine_id == 2].copy()
    x = engineer(train, ["sensor_1"], 3)
    model = make_pipeline(StandardScaler(), Ridge()).fit(x, targets(train))
    before = model[0].mean_.copy()
    val["sensor_1"] += 10000
    model.predict(engineer(val, ["sensor_1"], 3))
    assert (model[0].mean_ == before).all()
    assert model[0].mean_[1] == train.sensor_1.mean()
