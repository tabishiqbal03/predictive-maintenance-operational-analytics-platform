> Update, 7 October 2026: the project owner independently verified Docker build, container startup, health/model-info/predict and native/container parity, plus local CI-equivalent checks. Remote GitHub Actions remains unverified. Docker-unavailable statements below describe the earlier implementation snapshot and are superseded by this update. Version 3 and its verification history are preserved; see publication_readiness.md.

# Limitations and production considerations

- **External validity:** FD001 is simulated turbofan data, one operating condition and one
  degradation mode. It does not establish performance on real machinery, other fleets,
  maintenance interventions or changing operating conditions.
- **Labels:** training engines run to failure; deployed fleets include censored assets,
  repairs and replacement. Cycle-based RUL is not calendar time. Ground truth itself can
  depend on how failure is defined.
- **Evaluation:** one held-out-engine split and artificial validation truncations create
  selection uncertainty. No confidence interval or calibrated prediction interval is supplied.
  Test metrics describe 100 endpoints, not continuous online reliability. Rare misses can matter
  much more than aggregate RMSE. Long-life errors may dominate uncapped scoring.
- **Decision assumptions:** fixed horizons encode no actual maintenance capacity, parts lead
  times, costs or safety limits. Alerts are a human review queue. Early first alerts can be
  false alarms; retrospective warning lead time is not a causal intervention benefit.
- **Data quality:** invalid input fails fast. Real ingestion needs units, sensor calibration,
  missing/gapped history policies, time ordering, duplicates, schema versioning and provenance.
- **Drift:** monitor input distributions, missingness, alert rates and eventually delayed-label
  MAE/RMSE by asset cohort. Operating regime changes may invalidate the FD001 feature filter.
  Establish drift thresholds on deployment data; alert-rate changes alone do not prove model drift.
- **Retraining:** use versioned data and grouped/time-aware backtests, independent release gates,
  rollback and a frozen test policy. Avoid repeated tuning against this benchmark test set.
- **Security:** this local API has no authentication, TLS, request-rate controls or tenant
  isolation. Bind to localhost. Only load trusted joblib artifacts, which can execute code.
- **Scale:** SQLite suits a local portfolio demonstration, not concurrent high-rate sensor
  ingestion. Stream processing would require durable event handling and per-engine rolling state.
  A server database and monitoring should follow measured requirements, not precede them.
- **Availability:** no cloud deployment or live industrial integration is claimed. The API
  detects registry version changes without restart; do not modify immutable model files.
  Original benchmark generation still must not run concurrently with benchmark-page reads.
- **Impact:** no measured downtime, financial saving or causal operational benefit is available.
  A prospective monitored trial and engineering sign-off would be needed before real decisions.

## MLOps boundaries

The extension is an executed local lifecycle, not a managed industrial deployment. Artificial
monitoring reuses training engines and can have optimistic reference errors. Sensor/setting
perturbations are stress tests, not physically validated degradation mechanisms. Drift thresholds
and gates need prospective calibration. Reusing one validation split repeatedly risks overfitting
release decisions. The retraining example uses existing data and does not claim drift remediation.

MLflow artifacts and SQLite records are trusted local data; they are not signed, access-controlled
or tamper-proof. Checksums detect accidental corruption but do not protect against an attacker
who can modify both the artifact and registry. Back up the model files and both databases together.
Cross-store crash recovery and disaster recovery have not been tested. SQLite telemetry is best
effort and can be lost if persistence fails; there is no alert-delivery or incident-response system.

Real deployment would require managed compute/containers, durable object/model storage,
authentication and authorisation, secret management, scalable telemetry and alert routing,
production databases, scheduling/orchestration, fresh delayed-label handling, human approvals,
security review and tested backup/restore. These are considerations, not implemented claims.
Docker build and container inference parity were independently verified locally by the project owner.
GitHub Actions is configured but not remotely executed. MLflow tracking/UI work on Windows; its job
execution backend is unsupported there and is not part of this implementation.

