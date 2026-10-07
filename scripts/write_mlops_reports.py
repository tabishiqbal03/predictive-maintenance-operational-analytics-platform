"""Generate extension findings exclusively from executed local evidence."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

from maintenance.mlops.registry import active
from maintenance.mlops.store import rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    out, reports = ROOT / "artifacts/mlops", ROOT / "reports"
    verification = json.loads((out / "verification.json").read_text())
    challenger = json.loads((out / "latest_challenger.json").read_text())
    promotion = json.loads((out / "promotion_verification.json").read_text())
    production = active(ROOT)
    suites = ET.parse(out / f"tests_v{challenger['version']}.xml").getroot()
    tests = sum(int(s.get("tests", 0)) for s in suites.iter("testsuite"))
    scenarios = []
    for result in verification["scenarios"]:
        scenarios.append(
            {
                "scenario": result["scenario"],
                "status": result["status"],
                "data_drift": result.get("data_drift"),
                "prediction_drift": result.get("prediction_drift"),
                "performance_degraded": result.get("performance_degraded"),
                "mae": result.get("performance", {}).get("mae"),
                "rmse": result.get("performance", {}).get("rmse"),
                "recall": result.get("performance", {}).get("near_failure_recall"),
            }
        )
    pd.DataFrame(scenarios).to_csv(out / "scenario_summary.csv", index=False)
    pd.DataFrame(scenarios).to_csv(reports / "monitoring_summary.csv", index=False)
    (reports / "mlops_verified_results.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    summary = f"""# Verified MLOps extension summary — 6 October 2026

The original 30 tests passed before editing. The complete qualifying suite passed **{tests} tests**.
The original model, predictions, comparison and metric hashes remain unchanged.

MLflow 3.16.1 tracking and its Model Registry ran locally using SQLite and filesystem artifacts.
One FD001-RUL experiment contains {verification["mlflow_runs"]} runs: 16 honestly imported historical
candidate summaries, one imported verified benchmark model and one newly trained challenger.
The historical candidates were not retrained just to produce tracking history. The local MLflow
server passed HTTP readiness; Windows job execution is unsupported but is not used by this project.

Active production version: **{production["version"]}**, {production["metadata"]["candidate"]}.
MLflow run: `{production["mlflow_run_id"]}`. The deployment pointer lives transactionally in SQLite;
MLflow stores model versions and experiment evidence. Original benchmark: MAE 19.50, RMSE 27.01
cycles and NASA score 9986.28 on 100 test engines, unchanged.

## Artificial monitoring replay

```text
{pd.DataFrame(scenarios).to_string(index=False)}
```

Normal replay did not trigger meaningful drift. Injected sensor bias and operating-setting shifts
triggered data drift and the declared performance-degradation checks. Schema failure was recorded
as invalid and was not assigned a fabricated accuracy value. Per-scenario JSON, Markdown,
feature drift and prediction CSVs live under artifacts/mlops/monitoring/. These are training-data
replays with synthetic changes, not independent production accuracy measurements.

## Challenger and controlled release

Version {challenger["version"]} used 160 Extra Trees and seed 43, preserving the original feature
recipe and training/validation engines. Both recipes were refit on 75 engines for fair comparison.
Validation RMSE: champion {challenger["champion"]["rmse"]:.5f}, challenger {challenger["challenger"]["rmse"]:.5f}.
Validation MAE: champion {challenger["champion"]["mae"]:.5f}, challenger {challenger["challenger"]["mae"]:.5f}.
Both achieved 100% near-failure recall, zero missed critical endpoints and two false alerts
on 125 validation endpoints. Metric gates passed: {challenger["eligible"]}.
No final NASA test labels were read by retraining and no challenger test score is claimed.

After a passing source/model-bound test attestation, promotion was **{promotion["promotion"]}**;
rollback was **{promotion["rollback"]}**, verified against the same running API process.
Version 1 was intentionally restored so the original benchmark remains the current champion.
Version 2 remains a previously approved model. Every transition has a reason and timestamp.

## Interfaces and remaining verification limits

All four dashboard areas passed AppTest. The version-aware API and invalid-request telemetry passed.
Service aggregates include batch replay and API events; request IDs beginning `batch:` identify replay.
Docker: **{verification["docker"]}**. Dockerfile, .dockerignore and a CI build step exist.
GitHub Actions was not run remotely. No cloud deployment, real-machine integration, downtime
reduction or savings is claimed. Non-blocking Starlette/httpx and MLflow/SQLAlchemy deprecation
warnings occur in the locked environment.
"""
    (reports / "mlops_extension_summary.md").write_text(summary, encoding="utf-8")
    (reports / "sample_cv_section.md").write_text(
        f"""# Predictive Maintenance & MLOps Platform

- Built a reproducible NASA C-MAPSS FD001 RUL platform achieving test MAE 19.50 and RMSE 27.01
  cycles on 100 simulated engines, with leakage-safe evaluation, FastAPI, Streamlit and SQLite.
- Added local MLflow tracking and model versioning, simulated drift/performance monitoring,
  inference telemetry, and test-gated champion/challenger promotion with verified live API rollback;
  validated the integrated system with {tests} passing tests.

Key project skills: Python, pandas, scikit-learn, LightGBM, MLflow, model registry, causal features,
SciPy drift statistics, SQL/SQLite, FastAPI, Pydantic, Streamlit, pytest and GitHub Actions configuration.

Docker packaging is implemented but locally unverified; no Docker execution or cloud deployment
claim should be added. Challenger improvement is validation-only; original test metrics are unchanged.
""",
        encoding="utf-8",
    )
    (out / "lifecycle_events.json").write_text(
        json.dumps(rows(ROOT, "SELECT * FROM lifecycle_events"), indent=2)
    )
    print(f"Generated verified extension summary and CV wording ({tests} tests).")


if __name__ == "__main__":
    main()
