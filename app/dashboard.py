"""Read-only operational dashboard backed by the latest SQLite run."""

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(os.environ.get("MAINTENANCE_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "artifacts"
st.set_page_config(page_title="Fleet / Maintenance Analytics", page_icon="🛠", layout="wide")
st.title("Fleet maintenance intelligence")
st.caption("NASA C-MAPSS FD001 · Simulated turbofan engines · Decision support prototype")
st.caption(
    "Local prototype: no real company/customer data or industrial deployment; no measured downtime or financial impact."
)
if not (OUT / "maintenance.sqlite").exists():
    st.info("No scored fleet yet. Run `python run_pipeline.py --download` from the repository root.")
    st.stop()

else:
    # Keep the empty-workspace startup path free of chart-library imports.
    import altair as alt
    from maintenance.dashboard_views import PRIORITY, sensor_chart

with sqlite3.connect(f"file:{(OUT / 'maintenance.sqlite').as_posix()}?mode=ro", uri=True) as conn:
    fleet = pd.read_sql_query("SELECT * FROM current_predictions ORDER BY predicted_rul", conn)
    counts = pd.read_sql_query("SELECT * FROM fleet_status", conn)
    history = pd.read_sql_query("SELECT * FROM engine_cycles", conn)

page = st.sidebar.radio(
    "Workspace", ["Fleet Overview", "Asset Detail", "Model Performance", "MLOps / Monitoring"]
)
st.sidebar.caption("Action thresholds: Critical ≤20; Schedule ≤40; Monitor ≤80; Healthy >80 cycles.")
st.sidebar.caption("Illustrative planning assumptions. Predictions are not calibrated guarantees.")

if page == "Fleet Overview":
    st.subheader("Maintenance review queue")
    tiles = st.columns(6)
    tiles[0].metric("Monitored assets", len(fleet))
    for tile, label in zip(tiles[1:5], PRIORITY):
        tile.metric(label, int((fleet.maintenance_status == label).sum()))
    tiles[5].metric("Mean predicted RUL", f"{fleet.predicted_rul.mean():.1f} cycles")
    left, right = st.columns([1, 2])
    with left:
        st.altair_chart(
            alt.Chart(counts)
            .mark_bar(color="#147d92")
            .encode(
                x=alt.X(
                    "maintenance_status:N",
                    sort=PRIORITY,
                    title="Maintenance status",
                    axis=alt.Axis(labelAngle=-45, labelLimit=200),
                ),
                y=alt.Y("assets:Q", title="Assets"),
            ),
            width="stretch",
        )
    with right:
        st.dataframe(
            fleet[["engine_id", "cycle", "predicted_rul", "maintenance_status"]],
            hide_index=True,
            width="stretch",
        )
    st.download_button("Download scored fleet", fleet.to_csv(index=False), "fleet.csv", "text/csv")
elif page == "Asset Detail":
    engine = st.selectbox("Engine", sorted(fleet.engine_id.tolist()))
    st.caption(
        "Operational prediction view. Engine 1 remains the default; all engines are selectable, including known errors."
    )
    asset = fleet[fleet.engine_id == engine].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Predicted RUL", f"{asset.predicted_rul:.1f} cycles")
    b.metric("Maintenance status", asset.maintenance_status)
    c.metric("Latest observed cycle", int(asset.cycle))
    sensor = st.selectbox("Sensor", ["sensor_4", "sensor_11", "sensor_12", "sensor_7", "sensor_15"])
    trajectory = history[history.engine_id == engine].set_index("cycle")
    trend = trajectory[[sensor]].copy()
    trend["trailing 15-cycle mean"] = trend[sensor].rolling(15, min_periods=1).mean()
    st.altair_chart(sensor_chart(trend), width="stretch")
    st.caption(
        "Trailing smoothing shows observed behaviour; sensor units differ and trends are not failure diagnoses."
    )
    importance = pd.read_csv(OUT / "feature_importance.csv").head(10)
    st.subheader("Global model drivers")
    st.altair_chart(
        alt.Chart(importance)
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            x=alt.X("rmse_increase:Q", title="Increase in validation RMSE (cycles)"),
        ),
        width="stretch",
    )
    st.caption(
        "Validation permutation importance: RMSE increase after shuffling. Global evidence, not an explanation of this engine."
    )
    if st.checkbox("Show retrospective evaluation truth (not available operationally)"):
        st.warning("Evaluation only: NASA supplied endpoint truth. Future RUL is not an operational input.")
        st.metric("Actual endpoint RUL (evaluation only)", f"{asset.actual_rul:.0f} cycles")
        st.metric("Prediction error (predicted − actual)", f"{asset.error:.2f} cycles")
elif page == "MLOps / Monitoring":
    from maintenance.mlops.dashboard import render

    render(ROOT)
else:
    result = json.loads((OUT / "metrics.json").read_text())
    st.subheader("Final NASA test endpoints")
    for tile, key in zip(st.columns(3), ["mae", "rmse", "nasa_score"]):
        tile.metric(key.upper().replace("_", " "), f"{result['test'][key]:.2f}")
    left, right = st.columns(2)
    scatter = (
        alt.Chart(fleet)
        .mark_circle(size=45)
        .encode(
            x=alt.X("actual_rul:Q", title="Actual RUL (cycles)"),
            y=alt.Y("predicted_rul:Q", title="Predicted RUL (cycles)"),
            tooltip=["engine_id", "actual_rul", "predicted_rul"],
        )
    )
    upper = float(max(fleet.actual_rul.max(), fleet.predicted_rul.max()))
    identity = (
        alt.Chart(pd.DataFrame({"rul": [0, upper]}))
        .mark_line(strokeDash=[5, 5], color="gray")
        .encode(x="rul:Q", y="rul:Q")
    )
    left.altair_chart(scatter + identity, width="stretch")
    right.altair_chart(
        alt.Chart(fleet)
        .mark_bar()
        .encode(
            x=alt.X("error:Q", bin=alt.Bin(maxbins=20), title="Predicted − actual RUL (cycles)"),
            y=alt.Y("count():Q", title="Engines"),
        ),
        width="stretch",
    )
    st.warning(
        "Higher-RUL errors remain larger: MAE 24.26 cycles above actual RUL 80 versus 6.31 at ≤20. "
        "Missed near-failure engine 41: actual 18 cycles, predicted 69.37."
    )
    st.subheader("Model selection on held-out training engines")
    st.dataframe(pd.read_csv(OUT / "model_comparison.csv"), hide_index=True)
    st.subheader("Errors by true RUL")
    st.dataframe(pd.read_csv(OUT / "error_by_rul.csv"), hide_index=True)
    st.subheader("Operational performance")
    st.dataframe(pd.DataFrame([result["operational"]]), hide_index=True)
    st.dataframe(pd.read_csv(OUT / "threshold_sensitivity.csv"), hide_index=True)
    st.caption(
        "Near failure: actual RUL ≤20. Sensitivity varies only the prediction trigger; false-alert truth stays actual RUL >40. "
        "The default 40-cycle policy still has 0 false alerts among 21 alerts."
    )
    st.subheader("Exploratory validation warning lead times")
    st.dataframe(pd.DataFrame([result["validation_lead_time"]]), hide_index=True)
    with st.expander("Raw evaluation details"):
        st.json(result)
    st.caption(
        "Complete held-out trajectories only; first alerts may be premature. No measured downtime or savings."
    )
