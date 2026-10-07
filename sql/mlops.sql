-- Active local deployment (MLflow version and training run are immutable identifiers).
SELECT p.version, p.updated_at, v.mlflow_run_id, v.lifecycle, v.metadata
FROM production_model p JOIN model_versions v USING(version);
SELECT timestamp, model_version, scenario, status,
       json_extract(summary, '$.data_drift') AS data_drift,
       json_extract(summary, '$.prediction_drift') AS prediction_drift,
       json_extract(summary, '$.performance.rmse') AS rmse,
       json_extract(summary, '$.performance_degraded') AS performance_degraded
FROM monitoring_runs ORDER BY timestamp DESC;
SELECT * FROM service_summary;
SELECT * FROM inference_events WHERE status_code >= 400 ORDER BY timestamp DESC LIMIT 20;
SELECT version, champion_version, timestamp, eligible, evidence
FROM challenger_evaluations ORDER BY timestamp DESC;
SELECT * FROM lifecycle_events ORDER BY id DESC;
