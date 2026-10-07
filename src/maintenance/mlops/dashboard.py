import json

import pandas as pd
import streamlit as st

from maintenance.mlops.registry import active
from maintenance.mlops.store import initialize, rows
from maintenance.dashboard_views import monitoring_tables


def render(root):
    initialize(root)
    st.subheader("Production lifecycle and simulated monitoring")
    st.caption(
        "Artificial train_FD001 replay, not independent production/generalisation accuracy. Benchmark views remain frozen."
    )
    try:
        model = active(root)
    except FileNotFoundError:
        st.info("Run `python mlops.py bootstrap` to initialise the production registry.")
        return
    a, b, c = st.columns(3)
    a.metric("Active registry version", model["version"])
    b.metric("Model", model["metadata"].get("candidate"))
    c.metric("Feature recipe", model["metadata"].get("feature_version"))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "mlflow_run_id": model["mlflow_run_id"],
                    "approved_at": model["approved_at"],
                }
            ]
        ),
        hide_index=True,
    )
    if model["metadata"].get("benchmark"):
        st.dataframe(pd.DataFrame([model["metadata"]["benchmark"]]), hide_index=True)
    st.caption(
        "Benchmark metrics apply only to the imported benchmark model; challengers have validation evidence."
    )
    history = rows(root, "SELECT * FROM monitoring_runs ORDER BY timestamp DESC")
    if history:
        records = [json.loads(r["summary"]) for r in history]
        valid, invalid, comparison = monitoring_tables(records)
        st.subheader("Latest successfully evaluated monitoring run")
        st.caption(
            "Successfully evaluated means the batch passed schema validation and could be evaluated; "
            "it does not mean healthy or drift-free. Data, prediction and decision drift are separate "
            "from performance degradation."
        )
        if not valid.empty:
            latest = valid.iloc[0]
            degraded = latest.get("performance_degraded")
            if pd.isna(degraded):
                st.info("Performance degradation not assessed")
            elif degraded:
                st.error("Performance degradation detected")
            else:
                st.success("No performance degradation detected")
            fields = [
                c
                for c in [
                    "timestamp",
                    "scenario",
                    "model_version",
                    "data_drift",
                    "prediction_drift",
                    "decision_drift",
                    "performance_degraded",
                    "performance.mae",
                    "performance.rmse",
                ]
                if c in valid
            ]
            st.dataframe(valid[fields].head(1), hide_index=True)
            st.write("Drifted inputs: " + (", ".join(latest.get("drifted_raw", [])) or "None detected"))
            with st.expander("Raw evaluated-report detail"):
                st.json(next(r for r in records if r["status"] == "valid"))
        else:
            st.info("No successfully evaluated monitoring run is available yet.")
        st.subheader("Schema / validation failures")
        if not invalid.empty:
            fields = [
                c
                for c in ["timestamp", "scenario", "model_version", "status", "schema_error"]
                if c in invalid
            ]
            st.dataframe(invalid[fields], hide_index=True)
            with st.expander("Raw invalid-report evidence"):
                st.json([r for r in records if r["status"] != "valid"])
        else:
            st.caption("No recorded schema failures.")
        table = pd.json_normalize(records)
        cols = [
            c
            for c in [
                "timestamp",
                "scenario",
                "model_version",
                "status",
                "data_drift",
                "prediction_drift",
                "decision_drift",
                "performance_degraded",
                "performance.mae",
                "performance.rmse",
                "performance.near_failure_recall",
                "performance.false_alerts",
            ]
            if c in table
        ]
        st.subheader("Scenario comparison — latest evaluated run per scenario and model version")
        st.dataframe(comparison[[c for c in cols if c in comparison]], hide_index=True)
        with st.expander("Complete monitoring history (including invalid runs)"):
            st.dataframe(table[cols], hide_index=True)
        st.caption(
            "Scenarios are separate artificial experiments, not a chronological performance trend. Drift alone does not establish performance loss."
        )
    else:
        st.info("No monitoring batches yet. Use the simulate and monitor commands.")
    st.subheader("Inference service — API requests and labelled batch replay events")
    service = rows(root, "SELECT * FROM service_summary")
    st.dataframe(pd.DataFrame(service), hide_index=True)
    st.caption(
        "Latency excludes telemetry persistence. HTTP validation failures are recorded; no raw request bodies are stored."
    )
    st.subheader("Versions and MLflow runs")
    st.dataframe(
        pd.DataFrame(
            rows(root, "SELECT version,mlflow_run_id,created_at,approved_at,lifecycle FROM model_versions")
        ),
        hide_index=True,
    )
    st.subheader("Challenger evaluations")
    evaluations = rows(root, "SELECT * FROM challenger_evaluations ORDER BY timestamp DESC")
    for evaluation in evaluations:
        with st.expander(f"Version {evaluation['version']} against {evaluation['champion_version']}"):
            st.json(json.loads(evaluation["evidence"]))
    st.subheader("Promotion and rollback audit")
    st.dataframe(pd.DataFrame(rows(root, "SELECT * FROM lifecycle_events ORDER BY id DESC")), hide_index=True)
