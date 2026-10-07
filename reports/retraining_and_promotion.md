# Retraining, gates and release control

`python mlops.py retrain` uses the existing train_FD001 checksum, fitting-engine IDs and
validation-engine IDs. It does not load test_FD001 or RUL_FD001, monitoring labels or biased
replay data. This demonstrates reproducible retraining on existing data, not adaptation to a
new production population. Future real retraining would need approved fresh labels and a new
validation contract.

The production champion was fitted on all 100 training engines. Evaluating that estimator
directly on the 25 validation engines would leak. Instead, clone its recipe and fit it on
the original 75 engines; fit the challenger on exactly those engines too. Evaluate both at
five predetermined fractional-life endpoints per held-out engine. After comparison, fit a
deployment copy of the challenger on all training engines. This copy stays inactive.

The single declared challenger changes Extra Trees from 120 estimators/seed 42 to 160
estimators/seed 43, retaining the 15-cycle feature recipe, uncapped targets, leaf size and
feature sampling. This is a modest variance/capacity comparison, not a search tuned to win.

## Predeclared gates

- RMSE and MAE each <=102% of the champion recipe's validation score.
- Near-failure recall >=max(90%, champion recall minus 2 percentage points).
- No increase in missed critical endpoints; no more than one additional false alert.
- Finite metrics and a readable artifact whose checksum matches its registry entry.
- Passing complete pytest evidence bound to the source/test/dependency-lock hash and candidate
  artifact hash. Source or model changes invalidate the attestation; rerun `qualify`.
- Evaluation must refer to the currently active champion, preventing stale promotion after
  another model becomes production. The user supplies an audit reason for each release.

The 2% regression allowance recognises small-sample variation without claiming statistical
equivalence. Critical misses receive a stricter no-regression rule. These illustrative gates
were implemented before the challenger ran; they are not tuned using NASA test scores.
The policy accepts comparable models, not just winners. No automatic promotion follows training.

## Observed outcome

| Validation metric | Champion recipe | Challenger version 2 |
|---|---:|---:|
| MAE | 21.15763 | 20.94289 |
| RMSE | 29.23005 | 28.80403 |
| NASA sum | 8228.17 | 6026.69 |
| Near-failure recall | 25/25 | 25/25 |
| Missed critical endpoints | 0 | 0 |
| False alerts | 2 | 2 |

All metric gates passed. The improvement is validation-only, on an already-used selection
split; it is not a new independent benchmark result. The challenger has no claimed final-test
metrics. Evidence: `artifacts/mlops/latest_challenger.json` and its MLflow run artifacts.

## Promotion and rollback mechanics

MLflow registers immutable version identifiers and run artifacts. The local SQLite registry
stores relative deployment paths, SHA-256, recipe/version metadata and approval history.
`production_model` is the single deployment authority; MLflow aliases are deliberately not
maintained as a second competing pointer. Under `BEGIN IMMEDIATE`, gates/attestation/current
champion are checked and the pointer, version states and audit event commit together.
Files are copied and verified before a registry row is inserted; abandoned files after a crash
are possible, but an incomplete artifact is not intentionally activated.

FastAPI resolves the pointer at each request and caches bundles by version/checksum. An
in-flight prediction uses one model snapshot. Rollback accepts only previously approved models
and verifies their artifact. Local registry writes are trusted administrative actions, not a
security boundary or a tamper-proof approval system.

The executed demonstration qualified version 2, promoted 1→2, and rolled back 2→1 against
one live Uvicorn process. The restored prediction matched the original. Version 1 intentionally
remains production; version 2 remains approved/previous. Read
`artifacts/mlops/promotion_verification.json` and the `lifecycle_events` table for exact evidence.
