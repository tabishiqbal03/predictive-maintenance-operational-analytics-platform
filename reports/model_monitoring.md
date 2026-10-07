# Model monitoring

Monitoring is a local simulation using **train_FD001 only**. The original test files and
benchmark metrics are excluded. The deployed model has already seen these training engines;
the low replay error is in-sample and must not be reported as new generalisation accuracy.

## Replay construction

Seed 2026 chooses one endpoint between 20% and 95% of each of 100 training lifetimes.
Each input includes the available consecutive history up to that endpoint. Actual RUL is
derived before truncation. All scenarios compare the same engine endpoints against their
unperturbed histories to isolate the injected change; this is not independent sampling.

| Scenario | Artificial change |
|---|---|
| normal | Gaussian noise on sensors, SD = 2% of each sensor's training SD |
| sensor_bias | +3 training SD on sensors 4/11; −3 SD on sensors 7/12 throughout history |
| operating_shift | +3 training SD on settings 1/2; out-of-regime stress test, not a physically validated operating scenario |
| schema_failure | Remove sensor_11; persist an invalid-data report without scoring |

Bias scenarios emulate measurement/distribution changes with unchanged labels. They do not
simulate a physically different engine failure mechanism. Batch directories contain raw
histories, references, labels and a seed/scenario manifest. `monitor --unlabelled` omits all
performance conclusions when labels are unavailable.

## Three distinct signals

1. **Data drift:** raw channels (including dropped channels) and derived endpoint features.
   Flag only when empirical KS distance >0.2 **and** Wasserstein distance divided by
   reference SD >0.5. A 1e-9 floor handles constant reference channels. These are declared
   effect-size heuristics, not significance tests, calibrated industrial thresholds or
   multivariate guarantees. They use SciPy's open-source distribution statistics.
2. **Prediction and decision drift:** apply the same distribution rule to predicted RUL;
   measure status total-variation distance and action-rate change separately. Decision drift
   is flagged when status TV >0.1 or absolute alert-fraction change >0.1.
3. **Labelled performance:** MAE, RMSE, NASA sum, near-failure recall, false alerts and misses.
   Flag degradation when RMSE >max(1.2 × paired reference RMSE, reference RMSE +2 cycles),
   or near-failure recall drops by >5 percentage points. Null means not evaluated, not good.

SciPy was selected instead of adding Evidently because the small, explicit calculations fit
this project and avoid another report/schema abstraction. The implementation persists both
machine-readable JSON/CSV and readable Markdown, with test coverage for unchanged, shifted,
unlabelled and invalid data. A test demonstrates that drift in unused sensors can occur
without prediction or performance changes.

## Executed results on production version 1

| Scenario | Data drift | RUL distribution flag | MAE | RMSE | False action alerts |
|---|---|---|---:|---:|---:|
| Unperturbed paired reference | — | — | 4.00 | 5.56 | 0 |
| Normal | No | No | 4.52 | 6.23 | 0 |
| Sensor bias | Yes: sensors 4/7/11/12 | No | 21.95 | 31.38 | 7 |
| Operating shift | Yes: settings 1/2 | No | 8.48 | 12.84 | 0 |
| Schema failure | Invalid input | Not evaluated | — | — | — |

All three valid batches identified all 10 near-failure endpoints at the 40-cycle action
threshold. Both shift scenarios triggered labelled performance degradation; sensor bias
also triggered status-distribution change. **Neither shift crossed the declared global RUL
distribution threshold**, despite worse labelled errors. This is evidence that distribution
alerts alone can miss degradation. Do not change the thresholds merely to make every flag fire.

Sources: `reports/mlops_verified_results.json`, `reports/monitoring_summary.csv`, and per-run
`artifacts/mlops/monitoring/<id>/report.json`. SQLite `monitoring_runs` preserves scenario,
version, status, report path and summary; `sql/mlops.sql` gives history queries.

## Service monitoring

Prediction middleware records UTC timestamp, request ID, resolved model version, HTTP status,
latency, RUL/status on success and a generic error category on failure, including Pydantic 422s.
Raw request bodies are not logged. SQLite writes happen after response construction, so reported
latency excludes persistence. Telemetry failures are logged without hiding a successful prediction.
Batch replay events use request IDs beginning `batch:` and must be separated from API traffic
when interpreting volume/latency. `service_summary` is intentionally an aggregate across both.

References: [SciPy KS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
and [Wasserstein distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html).
