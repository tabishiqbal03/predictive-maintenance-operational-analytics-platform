from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_empty_dashboard(monkeypatch, tmp_path):
    monkeypatch.setenv("MAINTENANCE_ROOT", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/dashboard.py")).run()
    assert not app.exception
    assert "No scored fleet" in app.info[0].value
