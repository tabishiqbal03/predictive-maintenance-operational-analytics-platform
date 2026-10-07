# Technical decisions

FD001 keeps scope to one operating condition and one simulated high-pressure-compressor
degradation mode. Adding other subsets would require an explicit generalisation study.

## Labels and splits

Training RUL is each engine's final cycle minus its current cycle. Failure at the last
training cycle is a benchmark assumption. Test RUL is supplied for each final observed
test cycle, ordered by engine ID; we never substitute the end of a censored history for failure.
The seed-42 split uses 75 fitting engines and 25 validation engines. No engine ID is a feature.
Five fixed fractional-life endpoints per validation engine yield equal engine weighting and
include early-life uncertainty. Their distribution is a designed proxy, not the unknown
NASA test sampling scheme. Labels can use full run-to-failure history; features cannot.

## Features and model selection

A fitting-engine-only channel filter removes <=2 distinct values and channels with
relative range <=1e-5 (range divided by max(abs(mean), 1)). The final retained list stays
fixed during refitting on all training engines. Cycle, current channel values, trailing
mean, population standard deviation and one-cycle delta yield a small interpretable set.
Trailing windows of 5 and 15 cycles compare responsiveness with noise smoothing.
At startup, rolling statistics use available observations; the first delta is zero.
No backward fill, centered rolling operation, lifetime-normalised age or future slope is used.
API inference requires min(window, current cycle) consecutive recent observations.

Sixteen candidates compare four model families, two windows and two target formulations.
The mean predictor is deliberately naive. Ridge fits its scaler on fitting engines only.
Extra Trees and LightGBM use modest fixed capacity; this is a structured comparison,
not exhaustive hyperparameter optimisation. Uncapped and 125-cycle capped training labels
are compared using the same uncapped validation labels. A cap can represent uncertain
healthy life but can also bias early-life estimates down; it is not automatically preferred.
RMSE selects the winner; MAE and asymmetric NASA score provide complementary error views.
The selected model is refit on all training engines before the test files are loaded.
No decisions are revised using final-test metrics. Re-running is reproducibility, not new evidence.

Permutation importance is computed on validation endpoints before final refitting.
It measures predictive dependence under shuffling; correlated features complicate interpretation.
No SHAP or neural model was added: the classical pipeline already supports a clear analysis.

## Decisions and serving

Critical <=20, Schedule Maintenance <=40, Monitor <=80 and Healthy >80 are illustrative
cycle horizons, fixed before test evaluation. They are not standards or calibrated risks.
Sensitivity at 20/30/40/50/60 cycles shows the alert tradeoff without selecting a test-optimal threshold.
All paths share `maintenance.decision.status`; negative model output is clipped to zero.
Batch predictions include engine, cycle, UTC scoring time, truth/error and run ID.

SQLite is sufficient for a local, read-mostly fleet. Runs and endpoint predictions persist
historically; current cycle history is replaced per run. Views provide current status and
maintenance queues. Intermediate feature matrices remain CSV files, avoiding a duplicated
wide SQL feature table with no actual query use. Model/run files are archived by run ID.
The dashboard is a read-only SQL client; FastAPI loads the same trusted joblib bundle.
The bundle contains preprocessing where needed and the feature recipe, preventing serving skew.
The original direct-file cache has been superseded by the version-aware MLOps resolver described below.

## Reproducibility and limits

The dependency lock, input hashes, seed, split IDs, candidate table and metadata preserve
the experiment. Local snapshots are not a production model registry. One split is inexpensive
and auditable, but repeated grouped validation would better quantify selection variability.
SQLite population and artifact writes are not a single cross-file transaction; do not run
two pipelines concurrently or retrain while apps are reading. A future release process should
stage a run, validate it, and atomically promote a complete version.

## Implemented MLOps extension

MLflow now manages local experiments and registered model-version identifiers. A small SQLite
deployment abstraction supplies approval states, checksum-verified relative artifact paths,
test attestations and one active pointer. MLflow aliases are not duplicated; this avoids
ambiguous dual authority. Promotion is transactional for the pointer/history, while the
original pipeline's latest benchmark files remain a separate, non-atomic reporting workflow.

SciPy effect-size drift metrics were chosen over Evidently to keep the small monitoring
contract explicit and testable. All monitoring scenarios are training-data replay and
labelled artificial shifts. Thresholds are heuristics, not statistical significance or
industrial standards. Prediction distribution drift and measured performance loss are
stored separately; the executed scenarios demonstrate that their flags can disagree.

The deployed model has seen all training engines, so challenger evaluation reconstructs
the champion recipe on fitting engines rather than scoring its full-data fit on validation.
One modest tree-count/seed alternative is trained; the test set stays outside the workflow.
Documentation in retraining_and_promotion.md gives gate tolerances and their rationale.
The 48-test suite is bound to a source/model hash before promotion. Explicit CLI release
commands provide a local administrative decision, not enterprise RBAC or signed approvals.

