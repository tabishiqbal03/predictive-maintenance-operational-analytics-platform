"""Explicit artificial replay scenarios, effect-size drift and label-dependent performance."""

import json
from pathlib import Path
from time import perf_counter
import uuid

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from maintenance.data import COLUMNS, load, targets, validate
from maintenance.decision import status
from maintenance.evaluation import metrics, operational
from maintenance.features import engineer
from maintenance.model import load_model
from maintenance.mlops.registry import active, artifact
from maintenance.mlops.store import connection, now, write_json
from maintenance.mlops.telemetry import record_event


def simulate(root, scenario, seed=2026):
    if scenario not in ["normal", "sensor_bias", "operating_shift", "schema_failure"]:
        raise ValueError("Unknown simulated scenario")
    root = Path(root)
    train = load(root / "data/raw/train_FD001.txt")
    y = targets(train)
    rng = np.random.default_rng(seed)
    selected, labels = [], []
    for engine, group in train.groupby("engine_id"):
        end = int(rng.uniform(0.2, 0.95) * (len(group) - 1))
        selected.append(group.iloc[: end + 1])
        labels.append({"engine_id": int(engine), "actual_rul": float(y.loc[group.index[end]])})
    reference = pd.concat(selected, ignore_index=True)
    batch = reference.copy()
    if scenario == "normal":
        for col in COLUMNS[5:]:
            batch[col] += rng.normal(0, 0.02 * train[col].std(), len(batch))
    elif scenario == "sensor_bias":
        for col, direction in [("sensor_11", 1), ("sensor_4", 1), ("sensor_12", -1), ("sensor_7", -1)]:
            batch[col] += direction * 3 * train[col].std()
    elif scenario == "operating_shift":
        for col in ["setting_1", "setting_2"]:
            batch[col] += 3 * train[col].std()
    else:
        batch = batch.drop(columns="sensor_11")
    batch_id = uuid.uuid4().hex
    folder = root / "artifacts/mlops/batches" / batch_id
    folder.mkdir(parents=True)
    reference.to_csv(folder / "reference.csv", index=False)
    batch.to_csv(folder / "batch.csv", index=False)
    pd.DataFrame(labels).to_csv(folder / "labels.csv", index=False)
    write_json(
        folder / "manifest.json",
        {
            "id": batch_id,
            "scenario": scenario,
            "seed": seed,
            "source": "train_FD001 only; in-sample replay, not independent generalisation evidence",
            "artificial": True,
            "reference": "same engine endpoints before perturbation",
            "modification": {
                "normal": "independent Gaussian sensor noise, sigma=0.02 training SD",
                "sensor_bias": "+3 SD sensors 11/4; -3 SD sensors 12/7 (measurement bias, labels unchanged)",
                "operating_shift": "+3 SD settings 1/2; stress test outside FD001 operating regime",
                "schema_failure": "sensor_11 deliberately removed",
            }[scenario],
        },
    )
    return folder


def drift(reference, current):
    results = []
    for col in reference.columns:
        a, b = np.asarray(reference[col], dtype=float), np.asarray(current[col], dtype=float)
        distance = float(wasserstein_distance(a, b))
        scale = float(np.std(a))
        normalized = distance / max(scale, 1e-9)
        ks = float(ks_2samp(a, b).statistic)
        results.append(
            {
                "feature": col,
                "ks_distance": ks,
                "normalized_wasserstein": normalized,
                "drifted": bool(ks > 0.2 and normalized > 0.5),
            }
        )
    return pd.DataFrame(results)


def distribution(pred):
    return pd.Series(pred).map(status).value_counts(normalize=True).to_dict()


def monitor(root, folder, labelled=True):
    root, folder = Path(root), Path(folder)
    record = active(root)
    bundle = load_model(artifact(root, record))
    manifest = json.loads((folder / "manifest.json").read_text())
    run_id = uuid.uuid4().hex
    report_dir = root / "artifacts/mlops/monitoring" / run_id
    report_dir.mkdir(parents=True)
    result = {
        "id": run_id,
        "timestamp": now(),
        "model_version": record["version"],
        "batch_id": manifest["id"],
        "scenario": manifest["scenario"],
        "artificial": True,
        "source": manifest["source"],
        "status": "valid",
        "labelled": labelled,
    }
    try:
        reference = validate(pd.read_csv(folder / "reference.csv"))
        batch = validate(pd.read_csv(folder / "batch.csv"))
    except ValueError as exc:
        result.update(
            status="invalid",
            schema_error=str(exc),
            data_drift=None,
            prediction_drift=None,
            performance_degraded=None,
        )
    else:
        ref_last, last = reference.groupby("engine_id").tail(1), batch.groupby("engine_id").tail(1)
        xr = engineer(reference, bundle.columns, bundle.window).loc[ref_last.index]
        xb = engineer(batch, bundle.columns, bundle.window).loc[last.index]
        # Include raw constant/dropped channels too: monitoring must detect new variation in ignored inputs.
        raw_drift = drift(ref_last[COLUMNS[2:]], last[COLUMNS[2:]])
        feature_drift = drift(xr, xb)
        raw_drift.to_csv(report_dir / "raw_drift.csv", index=False)
        feature_drift.to_csv(report_dir / "feature_drift.csv", index=False)
        pr = np.maximum(0, bundle.model.predict(xr))
        predictions = []
        for engine, history in batch.groupby("engine_id"):
            started = perf_counter()
            item = bundle.latest(history)
            predictions.append(item["predicted_rul"])
            record_event(
                root,
                f"batch:{manifest['id']}:{engine}",
                record["version"],
                200,
                (perf_counter() - started) * 1000,
                item,
            )
        pb = np.array(predictions)
        prediction_drift = drift(pd.DataFrame({"rul": pr}), pd.DataFrame({"rul": pb})).iloc[0].to_dict()
        dr, db = distribution(pr), distribution(pb)
        tv = 0.5 * sum(abs(dr.get(k, 0) - db.get(k, 0)) for k in set(dr) | set(db))
        alert_delta = float((pb <= 40).mean() - (pr <= 40).mean())
        result.update(
            data_drift=bool(raw_drift.drifted.any() or feature_drift.drifted.any()),
            drifted_raw=raw_drift.loc[raw_drift.drifted, "feature"].tolist(),
            drifted_features=feature_drift.loc[feature_drift.drifted, "feature"].tolist(),
            prediction_drift=bool(prediction_drift["drifted"]),
            prediction_statistics=prediction_drift,
            reference_status_distribution=dr,
            status_distribution=db,
            status_total_variation=tv,
            alert_fraction_change=alert_delta,
            decision_drift=bool(tv > 0.1 or abs(alert_delta) > 0.1),
            performance_degraded=None,
        )
        scored = last[["engine_id", "cycle"]].copy()
        scored["predicted_rul"], scored["maintenance_status"] = pb, [status(p) for p in pb]
        if labelled:
            labels = pd.read_csv(folder / "labels.csv")
            if labels.engine_id.duplicated().any():
                raise ValueError("Duplicate monitoring label keys")
            truth = (
                last[["engine_id"]]
                .merge(labels, on="engine_id", how="left", validate="one_to_one")
                .actual_rul
            )
            if not np.isfinite(truth).all() or (truth < 0).any():
                raise ValueError("Monitoring labels missing or invalid")
            reference_metrics = {**metrics(truth, pr), **operational(truth, pr)}
            batch_metrics = {**metrics(truth, pb), **operational(truth, pb)}
            recall_drop = (
                reference_metrics["near_failure_recall"] is not None
                and batch_metrics["near_failure_recall"] < reference_metrics["near_failure_recall"] - 0.05
            )
            degraded = (
                batch_metrics["rmse"] > max(reference_metrics["rmse"] * 1.2, reference_metrics["rmse"] + 2)
                or recall_drop
            )
            result.update(
                reference_performance=reference_metrics,
                performance=batch_metrics,
                performance_degraded=bool(degraded),
            )
            scored["actual_rul"] = truth.to_numpy()
        scored.to_csv(report_dir / "predictions.csv", index=False)
    write_json(report_dir / "report.json", result)
    (report_dir / "report.md").write_text(
        "# Simulated monitoring report\n\nArtificial training-data replay; not an independent accuracy estimate.\n"
        "Data drift, decision changes and labelled performance are separate signals.\n"
        "Drift uses KS distance >0.2 AND Wasserstein / reference SD >0.5; no hypothesis-test significance claim.\n"
        "Performance trigger: RMSE >max(1.2×reference, reference+2) or recall drop >5 percentage points.\n\n"
        + "```json\n"
        + json.dumps(result, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    with connection(root) as db:
        db.execute(
            "INSERT INTO monitoring_runs VALUES (?,?,?,?,?,?,?)",
            (
                run_id,
                result["timestamp"],
                record["version"],
                manifest["scenario"],
                result["status"],
                report_dir.relative_to(root).as_posix(),
                json.dumps(result),
            ),
        )
    return result
