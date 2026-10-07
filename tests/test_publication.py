import pandas as pd
import pytest

from maintenance.dashboard_views import monitoring_tables, sensor_chart
from maintenance.evaluation import operational


@pytest.mark.parametrize("threshold,expected", [(20, 0), (30, 0), (40, 0), (50, 1), (60, 2)])
def test_sensitivity_keeps_truth_boundary_fixed(threshold, expected):
    result = operational([35, 45, 55], [25, 45, 55], threshold)
    assert result["false_alerts"] == expected


def test_latest_valid_and_scenario_comparison_preserve_errors():
    records = [
        dict(timestamp="2026-10-07", scenario="schema_failure", model_version="1", status="invalid"),
        dict(timestamp="2026-10-06", scenario="normal", model_version="1", status="valid"),
        dict(timestamp="2026-10-05", scenario="normal", model_version="1", status="valid"),
        dict(timestamp="2026-10-04", scenario="normal", model_version="2", status="valid"),
    ]
    valid, invalid, comparison = monitoring_tables(records)
    assert valid.iloc[0].timestamp == "2026-10-06"
    assert invalid.iloc[0].scenario == "schema_failure"
    assert len(comparison) == 2
    assert set(comparison.model_version) == {"1", "2"}


def test_sensor_chart_preserves_raw_values_and_labels_cropped_axis():
    frame = pd.DataFrame({"sensor_4": [1400.0, 1401.0]}, index=pd.Index([1, 2], name="cycle"))
    chart = sensor_chart(frame)
    assert chart.data.reading.tolist() == [1400.0, 1401.0]
    encoding = chart.to_dict()["encoding"]
    assert encoding["y"]["scale"]["zero"] is False
    assert "does not start at zero" in encoding["y"]["title"]
