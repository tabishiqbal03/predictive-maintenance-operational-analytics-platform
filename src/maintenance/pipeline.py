import argparse
import hashlib
import logging
import platform
import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from maintenance.data import COLUMNS, FILES, download, load, targets
from maintenance.database import populate
from maintenance.decision import status
from maintenance.evaluation import metrics, operational
from maintenance.features import engineer, informative
from maintenance.model import ModelBundle
from maintenance.reporting import eda, evaluation_plots, save_json, write_results


def candidates():
    return {
        "mean": DummyRegressor(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=120, min_samples_leaf=5, max_features=0.8, random_state=42, n_jobs=2
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=40,
            reg_lambda=2,
            random_state=42,
            n_jobs=2,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        ),
    }


def run(root: Path, fetch: bool = False):
    raw, out, reports = root / "data/raw", root / "artifacts", root / "reports"
    for folder in [out, reports, root / "data/processed"]:
        folder.mkdir(parents=True, exist_ok=True)
    if fetch:
        download(raw)
    train = load(raw / FILES[0])
    y = targets(train)
    fit_ids, val_ids = train_test_split(train.engine_id.unique(), test_size=0.25, random_state=42)
    fit, val = train[train.engine_id.isin(fit_ids)], train[train.engine_id.isin(val_ids)]
    columns = informative(fit)
    # Endpoint positions are predetermined independently of sensor measurements and test data.
    endpoints = []
    for _, group in val.groupby("engine_id"):
        endpoints.extend(group.index[(np.array([0.2, 0.4, 0.6, 0.8, 0.95]) * (len(group) - 1)).astype(int)])
    rows, fitted = [], {}
    for window in [5, 15]:
        x_fit, x_val = engineer(fit, columns, window), engineer(val, columns, window).loc[endpoints]
        for cap in [None, 125]:
            for name, model in candidates().items():
                key = f"{name}_w{window}_cap{cap}"
                logging.info("Fitting %s", key)
                model.fit(x_fit, y.loc[fit.index].clip(upper=cap) if cap else y.loc[fit.index])
                prediction = np.maximum(0, model.predict(x_val))
                rows.append(
                    {"candidate": key, "window": window, "cap": cap, **metrics(y.loc[endpoints], prediction)}
                )
                fitted[key] = ModelBundle(model, columns, window, {})
    comparison = pd.DataFrame(rows).sort_values("rmse", kind="stable")
    comparison.to_csv(out / "model_comparison.csv", index=False)
    winner = comparison.iloc[0]
    bundle = fitted[winner.candidate]
    validation_predictions = val[["engine_id", "cycle"]].copy()
    validation_predictions["actual_rul"] = y.loc[val.index]
    validation_predictions["predicted_rul"] = bundle.predict(val)
    validation_predictions.to_csv(out / "validation_predictions.csv", index=False)
    first = validation_predictions[validation_predictions.predicted_rul <= 40].groupby("engine_id").first()
    lead = {
        "engines": len(val_ids),
        "engines_alerted": len(first),
        "mean_first_alert_lead_cycles": float(first.actual_rul.mean()) if len(first) else None,
        "first_alerts_over_40_cycles": int((first.actual_rul > 40).sum()),
    }
    importance = permutation_importance(
        bundle.model,
        engineer(val, columns, bundle.window).loc[endpoints],
        y.loc[endpoints],
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=42,
        n_jobs=2,
    )
    pd.DataFrame(
        {
            "feature": engineer(fit, columns, bundle.window).columns,
            "rmse_increase": importance.importances_mean,
            "std": importance.importances_std,
        }
    ).sort_values("rmse_increase", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    created = datetime.now(UTC).isoformat()
    metadata = {
        "run_id": uuid.uuid4().hex,
        "created_at": created,
        "candidate": winner.candidate,
        "window": bundle.window,
        "cap": None if pd.isna(winner.cap) else float(winner.cap),
        "fit_engines": fit_ids.tolist(),
        "validation_engines": val_ids.tolist(),
        "seed": 42,
        "retained_channels": columns,
        "dropped_channels": [c for c in COLUMNS[2:] if c not in columns],
        "python": platform.python_version(),
        "training_sha256": hashlib.sha256((raw / FILES[0]).read_bytes()).hexdigest(),
        "selection_metric": "RMSE on 5 equal-weight endpoints per held-out engine",
    }
    save_json(out / "selection.json", metadata)  # Selection persisted before opening final test.
    bundle.metadata = metadata
    bundle.model.fit(
        engineer(train, columns, bundle.window), y.clip(upper=metadata["cap"]) if metadata["cap"] else y
    )
    joblib.dump(bundle, out / "best_model.joblib")
    test = load(raw / FILES[1])
    truth = pd.read_csv(raw / FILES[2], sep=r"\s+", header=None)
    if (
        truth.shape != (test.engine_id.nunique(), 1)
        or not np.isfinite(truth.to_numpy()).all()
        or (truth < 0).any().any()
    ):
        raise ValueError("RUL file must contain one finite nonnegative value per test engine")
    if sorted(test.engine_id.unique()) != list(range(1, len(truth) + 1)):
        raise ValueError("NASA truth requires contiguous test engine IDs starting at 1")
    last = test.groupby("engine_id").tail(1)
    all_pred = pd.Series(bundle.predict(test), index=test.index)
    pred = last[["engine_id", "cycle"]].copy()
    pred["predicted_rul"] = all_pred.loc[last.index]
    pred["actual_rul"] = truth.iloc[:, 0].to_numpy()
    pred["error"] = pred.predicted_rul - pred.actual_rul
    pred["maintenance_status"] = pred.predicted_rul.map(status)
    pred["run_id"], pred["prediction_timestamp"] = metadata["run_id"], created
    pred.to_csv(out / "predictions.csv", index=False)
    scores = metrics(pred.actual_rul, pred.predicted_rul)
    ops = operational(pred.actual_rul, pred.predicted_rul)
    sensitivity = [operational(pred.actual_rul, pred.predicted_rul, t) for t in [20, 30, 40, 50, 60]]
    pd.DataFrame(sensitivity).to_csv(out / "threshold_sensitivity.csv", index=False)
    bands = []
    for name, lo, hi in [("0-20", -1, 20), ("21-40", 20, 40), ("41-80", 40, 80), (">80", 80, np.inf)]:
        subset = pred[(pred.actual_rul > lo) & (pred.actual_rul <= hi)]
        if len(subset):
            bands.append({"rul_band": name, **metrics(subset.actual_rul, subset.predicted_rul)})
    bands = pd.DataFrame(bands)
    bands.to_csv(out / "error_by_rul.csv", index=False)
    save_json(out / "metrics.json", {"test": scores, "operational": ops, "validation_lead_time": lead})
    metadata["test_sha256"] = {f: hashlib.sha256((raw / f).read_bytes()).hexdigest() for f in FILES[1:]}
    save_json(out / "metadata.json", metadata)
    save_json(
        out / "validation.json",
        {
            "train_rows": len(train),
            "test_rows": len(test),
            "missing_values": 0,
            "duplicate_keys": 0,
            "invalid_records": 0,
            "checks": "26 numeric finite columns, positive integer keys, consecutive cycles from 1; fail-fast",
        },
    )
    train.assign(actual_rul=y).to_csv(root / "data/processed/train.csv", index=False)
    test.to_csv(root / "data/processed/test.csv", index=False)
    train[["engine_id"]].join(engineer(train, columns, bundle.window)).to_csv(
        root / "data/processed/train_features.csv", index=False
    )
    summary = eda(train, y, columns, reports)
    evaluation_plots(pred, reports)
    populate(out / "maintenance.sqlite", test, pred, metadata)
    write_results(root, metadata, scores, ops, summary, comparison, bands, lead)
    sample = test[test.engine_id == 1].tail(bundle.window)
    save_json(out / "example_request.json", {"observations": sample.to_dict(orient="records")})
    save_json(out / "example_response.json", bundle.latest(sample))
    # Preserve per-run evidence while latest files stay convenient for local apps.
    import shutil

    archive = out / "runs" / metadata["run_id"]
    archive.mkdir(parents=True)
    for file in out.iterdir():
        if file.is_file() and file.suffix in [".csv", ".json", ".joblib"]:
            shutil.copy2(file, archive / file.name)
    logging.info("Completed %s: %s", winner.candidate, scores)
    from maintenance.mlops.lifecycle import track_pipeline

    version = track_pipeline(root)
    logging.info("Tracked and registered version %s as candidate; production is unchanged", version)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate simulated NASA FD001")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(args.root, args.download)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Pipeline stopped: {exc}\n")
