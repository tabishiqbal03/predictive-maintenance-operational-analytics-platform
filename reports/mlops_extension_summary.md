> Update, 7 October 2026: the project owner independently verified Docker build, container startup, health/model-info/predict and native/container parity, plus local CI-equivalent checks. Remote GitHub Actions remains unverified. Docker-unavailable statements below describe the earlier implementation snapshot and are superseded by this update. Version 3 and its verification history are preserved; see publication_readiness.md.

# Verified MLOps extension summary — 6 October 2026

The original 30 tests passed before editing. The complete qualifying suite passed **48 tests**.
The original model, predictions, comparison and metric hashes remain unchanged.

MLflow 3.16.1 tracking and its Model Registry ran locally using SQLite and filesystem artifacts.
One FD001-RUL experiment contains 18 runs: 16 honestly imported historical
candidate summaries, one imported verified benchmark model and one newly trained challenger.
The historical candidates were not retrained just to produce tracking history. The local MLflow
server passed HTTP readiness; Windows job execution is unsupported but is not used by this project.

Active production version: **1**, extra_trees_w15_capNone.
MLflow run: `d2a9d7a136634901944ec4391b656bb0`. The deployment pointer lives transactionally in SQLite;
MLflow stores model versions and experiment evidence. Original benchmark: MAE 19.50, RMSE 27.01
cycles and NASA score 9986.28 on 100 test engines, unchanged.

## Artificial monitoring replay

```text
       scenario  status data_drift prediction_drift performance_degraded       mae      rmse  recall
         normal   valid      False            False                False  4.520694  6.230617     1.0
    sensor_bias   valid       True            False                 True 21.950274 31.380057     1.0
operating_shift   valid       True            False                 True  8.479504 12.838755     1.0
 schema_failure invalid       None             None                 None       NaN       NaN     NaN
```

Normal replay did not trigger meaningful drift. Injected sensor bias and operating-setting shifts
triggered data drift and the declared performance-degradation checks. Schema failure was recorded
as invalid and was not assigned a fabricated accuracy value. Per-scenario JSON, Markdown,
feature drift and prediction CSVs live under artifacts/mlops/monitoring/. These are training-data
replays with synthetic changes, not independent production accuracy measurements.

## Challenger and controlled release

Version 2 used 160 Extra Trees and seed 43, preserving the original feature
recipe and training/validation engines. Both recipes were refit on 75 engines for fair comparison.
Validation RMSE: champion 29.23005, challenger 28.80403.
Validation MAE: champion 21.15763, challenger 20.94289.
Both achieved 100% near-failure recall, zero missed critical endpoints and two false alerts
on 125 validation endpoints. Metric gates passed: True.
No final NASA test labels were read by retraining and no challenger test score is claimed.

After a passing source/model-bound test attestation, promotion was **passed**;
rollback was **passed**, verified against the same running API process.
Version 1 was intentionally restored so the original benchmark remains the current champion.
Version 2 remains a previously approved model. Every transition has a reason and timestamp.

## Interfaces and remaining verification limits

All four dashboard areas passed AppTest. The version-aware API and invalid-request telemetry passed.
Service aggregates include batch replay and API events; request IDs beginning `batch:` identify replay.
Docker: **not installed; build not verified**. Dockerfile, .dockerignore and a CI build step exist.
GitHub Actions was not run remotely. No cloud deployment, real-machine integration, downtime
reduction or savings is claimed. Non-blocking Starlette/httpx and MLflow/SQLAlchemy deprecation
warnings occur in the locked environment.
