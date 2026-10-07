# Project summary

This local decision-support platform estimates remaining useful cycles for simulated NASA
C-MAPSS FD001 turbofan engines. A modular Python pipeline validates raw trajectories, derives
run-to-failure labels, constructs causal history features, compares classical regressors on
held-out engines, refits the selected candidate and evaluates untouched NASA test endpoints.

Persisted predictions become a maintenance review queue through one shared threshold function.
SQLite stores model runs and predictions and exposes operational views. A three-area Streamlit
dashboard shows the fleet, individual histories and evaluation evidence. FastAPI accepts recent
engine histories through a validated schema and returns a RUL estimate, status and version.

The experiment saves raw-file hashes, split IDs, model/preprocessing bundle, comparisons,
operational metrics, threshold sensitivity, validation importance, EDA charts and reports.
Tests exercise schema failures, RUL derivation, feature causality, history equivalence, thresholds,
model serialization, API requests and SQL run selection without needing NASA data in CI.

See model_evaluation.md for generated results, business_findings.md for actual observed patterns,
technical_decisions.md for tradeoffs and ../HANDOFF.md for verified execution status.
This is a reproducible portfolio system, not an industrial safety system or a demonstrated
downtime-reduction intervention.

## MLOps extension, 6 October 2026

The same system now logs experiments to local MLflow and registers versioned model bundles.
A transactional SQLite production pointer supports version-aware inference, controlled
promotion and rollback. A fourth dashboard area displays artificial batch drift, labelled
performance, inference latency/errors and release history. Retraining compares champion
and challenger recipes on the existing fitting/validation engines, without reading test labels.
Promotion requires metric gates, artifact integrity and source/model-bound passing tests.
The original benchmark remains unchanged; see mlops_extension_summary.md for executed results.
