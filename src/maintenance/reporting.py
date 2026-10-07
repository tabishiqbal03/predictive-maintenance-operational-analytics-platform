import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def eda(frame, y, columns, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    profile = frame.describe().T
    profile["unique"] = frame.nunique()
    profile.to_csv(out / "sensor_profile.csv")
    life = frame.groupby("engine_id").cycle.max()
    corr = frame[columns].corrwith(y).sort_values()
    corr.rename("correlation_with_rul").to_csv(out / "sensor_rul_correlations.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(life, bins=18, color="#147d92")
    ax.set(xlabel="Engine lifetime (cycles)", ylabel="Training engines", title="FD001 lifetime variability")
    fig.tight_layout()
    fig.savefig(out / "lifetimes.png", dpi=140)
    plt.close(fig)
    sensors = [c for c in columns if c.startswith("sensor")]
    chosen = corr[sensors].abs().nlargest(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.flat, chosen):
        for engine in frame.engine_id.unique()[:5]:
            rows = frame.engine_id == engine
            ax.plot(y[rows], frame.loc[rows, col], alpha=0.6, linewidth=0.8)
        ax.set(title=col, xlabel="Actual cycles until failure")
        ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out / "degradation.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(frame[sensors].corr(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(sensors)), sensors, rotation=90)
    ax.set_yticks(range(len(sensors)), sensors)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "correlations.png", dpi=140)
    plt.close(fig)
    phases = frame[columns].assign(
        phase=y.map(lambda r: "late (<=20)" if r <= 20 else "healthy (>80)" if r > 80 else "middle")
    )
    phases.groupby("phase").mean().to_csv(out / "phase_means.csv")
    summary = {
        "engines": len(life),
        "rows": len(frame),
        "lifetime_min": int(life.min()),
        "lifetime_median": float(life.median()),
        "lifetime_max": int(life.max()),
        "retained_channels": columns,
        "strongest_absolute_correlations": {c: float(corr[c]) for c in chosen},
    }
    save_json(out / "eda_summary.json", summary)
    return summary


def evaluation_plots(predictions, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(predictions.actual_rul, predictions.predicted_rul, s=20, color="#147d92")
    top = max(predictions.actual_rul.max(), predictions.predicted_rul.max())
    axes[0].plot([0, top], [0, top], "--", color="gray")
    axes[0].set(xlabel="Actual RUL", ylabel="Predicted RUL", title="Final FD001 endpoints")
    axes[1].hist(predictions.error, bins=20, color="#147d92")
    axes[1].set(xlabel="Predicted − actual RUL", ylabel="Engines", title="Positive error means late warning")
    fig.tight_layout()
    fig.savefig(out / "test_errors.png", dpi=140)
    plt.close(fig)


def write_results(root, metadata, scores, operational, summary, comparison, bands, lead):
    reports = root / "reports"
    model = metadata["candidate"]
    predictions = pd.read_csv(root / "artifacts/predictions.csv")
    worst = predictions.loc[
        predictions.error.abs().nlargest(5).index,
        ["engine_id", "cycle", "actual_rul", "predicted_rul", "error"],
    ]
    missed = predictions[(predictions.actual_rul <= 20) & (predictions.predicted_rul > 40)]
    importance = pd.read_csv(root / "artifacts/feature_importance.csv").head(5)
    phases = pd.read_csv(reports / "phase_means.csv", index_col="phase")
    text = f"""# Model evaluation

Selected **{model}** using lowest engine-balanced validation RMSE, before reading NASA test labels.
The fixed split holds out 25 engines. Five deterministic truncation points per held-out engine
(20%, 40%, 60%, 80%, 95% of observed life) cover early and late operation equally by engine.
This proxy endpoint distribution differs from NASA test truncation; scores are not directly comparable.
Full validation trajectories are used only for retrospective warning analysis.

## Validation candidates (raw, uncapped RUL evaluation)

```text
{comparison.to_string(index=False)}
```

## Final test: one endpoint per engine, uncapped ground truth

```json
{json.dumps(scores, indent=2)}
```

NASA score sums exp(-error/13)-1 for negative errors and exp(error/10)-1 otherwise,
where error = predicted − actual. Lower is better; overprediction costs more. Its scale depends on sample count.
All predictions are clipped at zero. Training caps are compared against uncapped training targets.

## Operational evaluation

Near failure means actual RUL <=20; an actionable warning means predicted RUL <=40.
False alerts mean predicted <=40 and actual >40. These are project assumptions, not industry standards.
```json
{json.dumps(operational, indent=2)}
```

## Errors by true RUL band
```text
{bands.to_string(index=False)}
```

## Retrospective validation warning analysis
```json
{json.dumps(lead, indent=2)}
```
Lead time is actual cycles remaining at the first <=40 prediction on a held-out complete trajectory.
It is conditional on any alert, can count an early false alarm, and does not measure downtime avoided.
It comes from the selection validation set and is therefore exploratory. NASA test trajectories are
censored and cannot establish actual intervention lead time. Threshold sensitivity is saved separately;
thresholds were not tuned on test results. Feature permutation importance uses validation endpoints,
with correlated features potentially sharing or masking importance; it is not a causal explanation.
"""
    (reports / "model_evaluation.md").write_text(text, encoding="utf-8")
    with (reports / "model_evaluation.md").open("a", encoding="utf-8") as report:
        report.write(f"""
## Specific failures and interpretation

Largest absolute endpoint errors:
```text
{worst.to_string(index=False)}
```
Missed near-failure assets under the action rule:
```text
{missed[["engine_id", "actual_rul", "predicted_rul", "maintenance_status"]].to_string(index=False)}
```
The >80-cycle group has MAE {bands.loc[bands.rul_band == ">80", "mae"].iloc[0]:.2f},
compared with {bands.loc[bands.rul_band == "0-20", "mae"].iloc[0]:.2f} at <=20 cycles.
Early-life ambiguity is a material limitation. Large overestimates dominate the exponential NASA
score even when near-failure recall is high. The model was selected on RMSE, not the NASA score;
these objectives need not choose the same candidate. No model choice was revised after this analysis.

Top validation permutation drivers (RMSE increase in cycles):
```text
{importance.to_string(index=False)}
```
A strong age contribution is consistent with useful lifetime-distribution information, but
could be brittle if deployed engine lifetimes shift. Sensor importance is distributed across
correlated raw and smoothed channels. Negative or small values should not be read as precise rankings.
""")
    findings = f"""# Verified business findings

Training data contains {summary["engines"]} simulated engines and {summary["rows"]} observations.
Lifetimes range from {summary["lifetime_min"]} to {summary["lifetime_max"]} cycles
with median {summary["lifetime_median"]}. Age alone cannot capture all lifetime variation.

The largest absolute sensor/RUL Pearson correlations in training are
{summary["strongest_absolute_correlations"]}. These pooled associations are not causal and
do not account for repeated observations. See sensor_profile.csv, phase_means.csv,
degradation.png and correlations.png for distributions, phase means and redundancy.

Retained channels: {", ".join(summary["retained_channels"])}.
The training-only relative-range filter removes channels with <=2 values or relative range <=1e-5;
the complete dropped list is in artifacts/metadata.json. Retained operating settings may still be noise;
their utility should be assessed through a future grouped ablation.

Final test actionable alerts: {operational["alerts"]}; missed near-failure engines:
{operational["missed_critical"]}; false alerts: {operational["false_alerts"]}.
Operational precision is limited by the assumed 40-cycle planning horizon. These findings support
ranking a review queue, not automatic shutdowns or measured financial savings.
"""
    (reports / "business_findings.md").write_text(findings, encoding="utf-8")
    with (reports / "business_findings.md").open("a", encoding="utf-8") as report:
        report.write(f"""
## Observed phase differences and fleet composition

Sensor 11 means: healthy {phases.loc["healthy (>80)", "sensor_11"]:.3f},
late life {phases.loc["late (<=20)", "sensor_11"]:.3f}.
Sensor 12 means: healthy {phases.loc["healthy (>80)", "sensor_12"]:.3f},
late life {phases.loc["late (<=20)", "sensor_12"]:.3f}.
These pooled means support degradation association, not independence or a causal mechanism.

```text
{predictions.maintenance_status.value_counts().to_string()}
```
Cycle is the leading permutation driver in this run. The large errors listed in model_evaluation.md
show why apparently healthy classifications should not be treated as guarantees. Setting 1/2
vary slightly despite FD001's single operating regime; small importance differences could reflect
sample noise. A future setting-feature ablation belongs on fresh validation splits.
""")
    (reports / "sample_cv_section.md").write_text(
        f"""# Predictive Maintenance & Operational Analytics Platform

- Built a reproducible NASA C-MAPSS FD001 RUL pipeline with engine-separated model selection,
  FastAPI inference, Streamlit decision support and SQLite analytics; selected {model}, achieving
  final-test MAE {scores["mae"]:.2f} and RMSE {scores["rmse"]:.2f} cycles on {scores["n"]} simulated engines.

Key project skills: Python, pandas, scikit-learn, LightGBM, time-series feature engineering,
leakage prevention, regression evaluation, SQL, FastAPI, Streamlit, pytest.
""",
        encoding="utf-8",
    )
