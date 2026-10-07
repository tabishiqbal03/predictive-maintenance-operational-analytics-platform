# Publication readiness — 7 October 2026

This is a targeted dashboard, sensitivity-definition and documentation cleanup. No models
were fitted, registry versions created, lifecycle transitions performed or historical evidence
deleted. Version 1 stays production. Version 3 and its independent qualification/promotion/
rollback evidence remain preserved. Source changes intentionally make old qualification hashes
historical; obtain a new qualification before a future promotion, without treating old test
results as tests of this revision. This pass does not overwrite tests_v2/tests_v3 attestations.

## Corrected presentation

- Sensitivity varies the prediction/action threshold only. False alerts always require actual
  RUL >40, so the 40-cycle headline policy is unchanged. Only the current sensitivity CSV is
  regenerated from saved endpoint predictions; historical run archives retain their original
  definitions and must not be read as the corrected sensitivity table.
- Latest valid monitoring gets the main summary. Invalid/schema-failure records remain in
  a dedicated section and the complete history. Scenarios use a comparison table, not a line
  suggesting chronological deterioration; latest per scenario AND model version avoids mixing versions.
- Raw sensor chart axes adapt to the observed range and explicitly state that zero is excluded.
  Values are unchanged. Engine 1 remains default; optional truth is labelled retrospective only.
- Fleet counts follow Critical, Schedule Maintenance, Monitor, Healthy. Mean is labelled predicted.
- Permutation importance axis is increase in validation RMSE (cycles), not percentage/causality.
- App-native scatter/error charts use the existing saved predictions and Streamlit theme;
  the original report PNGs remain untouched. Operational and monitoring summaries use tables;
  raw detail remains available in expanders.
- The app retains simulation/threshold/impact caveats and explicitly shows high-RUL errors and
  missed engine 41. No independent production accuracy is claimed for monitoring replay.

## Paths and publication contents

Historical MLflow artifact locations are absolute `file:///C:/Users/tabis/...` URIs. They are
preserved, not edited in the tracking database. Nineteen finished runs and versions 1/2/3 remain.
The existing code already derives tracking paths from the supplied repository root and saves
deployment model paths relative to that root. A fresh clone can create its own local store
through the documented pipeline/bootstrap commands. Copying the historical tracking database
to a different location does not relocate its artifacts: retain the original location for
inspection or plan a backed-up, separately verified migration. Do not rewrite historical URIs
as an aesthetic cleanup. The API's relative deployment paths are independent of those URIs.

Keep source, tests, lockfile, SQL, README, Dockerfile, reports, small numerical benchmark
artifacts and charts in a GitHub export. Existing .gitignore excludes local environments,
caches, raw downloads, processed tables, model binaries, SQLite databases, run archives and
the full local MLOps store. Additional system/temp/secrets/build and local test/snapshot ignores
were added. .dockerignore already uses a narrow source/API/lockfile allowlist and was retained.

**No working-copy caches or evidence were deleted.** Excluded files are still necessary for
the existing working demo and should be backed up, especially both SQLite databases together
with the complete artifacts/mlops and artifacts/runs directories. Rebuilding training outputs
is possible; exact historical audit events and attestations are not reconstructed by training.
A source-only clone must install with `uv sync --frozen --extra dev`, obtain the three FD001
files, run the pipeline, then bootstrap. It will have new run IDs and timestamps. Do not zip
the entire working directory: .gitignore is not automatically honoured by ordinary ZIP tools.
No Git repository was initialised and nothing was pushed.

## Verification provenance

The user independently verified Docker build/start, container health/model-info/predict,
Version 1 loading and native/container prediction parity before this cleanup. The user also
verified local CI-equivalent dependency sync, Ruff, 48 tests and Docker build. These replace
the earlier reports' dated Docker-unavailable notes. Remote GitHub Actions remains unverified.
This pass does not claim a new Docker build. Publication-pass checks and exact changed files
are listed in the final response and HANDOFF.md.

## Manual dashboard recheck

1. Fleet: verify ordered statuses and counts 13/8/22/57 and the Mean predicted RUL label.
2. Asset Detail: select engines 1 and 41; inspect visible raw sensor variation and the axis
   disclosure. Toggle retrospective truth and verify it is clearly outside operational inputs.
3. Drivers: verify the RMSE-increase-in-cycles axis and global/noncausal explanation.
4. Model Performance: inspect chart contrast in light/dark themes, unchanged headline metrics,
   error limitations and fixed-truth sensitivity counts. Raw details should remain expandable.
5. MLOps: verify latest valid summary, separate schema_failure evidence, per-scenario/version
   comparison, complete history and versions 1/2/3 with Version 1 production.


## Final check results and file manifest

Final pytest: 55 passed, 0 failed (48 existing + 5 parameterised threshold checks,
1 latest-valid/scenario-history check and 1 raw-sensor chart check). Ruff: All checks passed!
Four dashboard areas and the truth toggle passed AppTest. Native API health/model-info confirmed
Version 1. MLflow server readiness passed and client inspection returned 19 finished runs and
versions 1/2/3. Protected file hashes and registry/tracking rows matched the before snapshot.
Existing Starlette/httpx and SQLAlchemy warnings remain non-blocking. No remote CI or new Docker
build was claimed. The first suite attempt hit the original empty-dashboard 3-second startup
limit; deferred chart imports fixed the startup path without changing tests or their limits.

Modified project files:
- .gitignore
- README.md
- HANDOFF.md
- app/dashboard.py
- src/maintenance/evaluation.py
- src/maintenance/mlops/dashboard.py
- artifacts/threshold_sensitivity.csv
- reports/production_architecture.md
- reports/limitations_and_production_notes.md
- reports/mlops_extension_summary.md
- reports/mlops_interview_notes.md
- reports/sample_cv_section.md

New project files:
- src/maintenance/dashboard_views.py
- tests/test_publication.py
- scripts/publication_audit.py
- reports/publication_readiness.md

Generated verification evidence:
- artifacts/publication_before.json (new, ignored local preservation snapshot)
- artifacts/publication_tests.xml (new, ignored final test output)
- artifacts/server_mlflow.log (refreshed, ignored server smoke log)

Routine ignored Python bytecode and Ruff cache entries were refreshed by checks. No runtime
folders, databases, models, historical monitoring batches or attestations were removed.
.dockerignore, Dockerfile, API, registry/tracking implementation, training pipeline and all
48 pre-existing tests were left unchanged.
