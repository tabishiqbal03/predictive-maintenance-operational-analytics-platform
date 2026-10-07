"""Training integration, honest benchmark import and train-only challenger evaluation."""

import json
from pathlib import Path
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from maintenance.data import load, targets
from maintenance.evaluation import metrics, operational
from maintenance.features import engineer
from maintenance.model import ModelBundle, load_model
from maintenance.mlops.registry import active, artifact, get_version, register, switch
from maintenance.mlops.store import connection, digest, initialize, now, rows, source_hash, write_json
from maintenance.mlops.tracking import log_run

FEATURE_VERSION = "causal-rolling-v1"


def endpoints(frame):
    result = []
    for _, group in frame.groupby("engine_id"):
        result.extend(group.index[(np.array([0.2, 0.4, 0.6, 0.8, 0.95]) * (len(group) - 1)).astype(int)])
    return result


def track_pipeline(root):
    """Import saved evidence honestly; also called after future end-to-end training runs."""
    root = Path(root)
    out = root / "artifacts"
    bundle = load_model(out / "best_model.joblib")
    meta = json.loads((out / "metadata.json").read_text())
    scores = json.loads((out / "metrics.json").read_text())
    comparison = pd.read_csv(out / "model_comparison.csv")
    initialize(root)
    existing = rows(root, "SELECT version,metadata FROM model_versions")
    for row in existing:
        if json.loads(row["metadata"]).get("source_run_id") == meta["run_id"]:
            return row["version"]
    manifest = {
        name: digest(out / name)
        for name in [
            "best_model.joblib",
            "metrics.json",
            "predictions.csv",
            "model_comparison.csv",
            "metadata.json",
        ]
    }
    write_json(out / "mlops" / ("benchmark_manifest_" + meta["run_id"] + ".json"), manifest)
    for row in comparison.to_dict("records"):
        log_run(
            root,
            "benchmark-evidence-" + row["candidate"],
            {
                "candidate": row["candidate"],
                "window": row["window"],
                "cap": row["cap"],
                "seed": 42,
                "feature_version": FEATURE_VERSION,
                "source_run_id": meta["run_id"],
            },
            {f"validation_{key}": row[key] for key in ["mae", "rmse", "nasa_score"]},
            tags={"evidence_type": "imported persisted comparison; no candidate refit"},
        )
    selected = comparison.iloc[0].to_dict()
    val = pd.read_csv(out / "validation_predictions.csv")
    val = val.loc[endpoints(val)]
    val_ops = operational(val.actual_rul, val.predicted_rul)
    metadata = {
        **meta,
        "source_run_id": meta["run_id"],
        "feature_version": FEATURE_VERSION,
        "benchmark": scores["test"],
        "validation": {k: selected[k] for k in ["mae", "rmse", "nasa_score"]},
        "validation_operational": val_ops,
        "imported_benchmark": True,
    }
    run_id = log_run(
        root,
        "verified-benchmark-model",
        {
            "model_type": type(bundle.model).__name__,
            "parameters": bundle.model.get_params(),
            "window": bundle.window,
            "channels": bundle.columns,
            "feature_version": FEATURE_VERSION,
            "target": meta["cap"],
            "seed": 42,
            "training_time": meta["created_at"],
            "training_sha256": meta["training_sha256"],
        },
        {
            **{f"test_{k}": v for k, v in scores["test"].items()},
            **{f"validation_{k}": v for k, v in metadata["validation"].items()},
            **{f"operational_{k}": v for k, v in val_ops.items()},
        },
        [
            out / name
            for name in ["model_comparison.csv", "metrics.json", "feature_importance.csv", "metadata.json"]
        ]
        + list((root / "reports").glob("*.png")),
        tags={"evidence_type": "verified persisted benchmark"},
    )
    return register(root, out / "best_model.joblib", run_id, metadata)


def bootstrap(root):
    version = track_pipeline(root)
    try:
        active(root)
    except FileNotFoundError:
        switch(root, version, "Initial import of verified FD001 benchmark", bootstrap=True)
    return version


def gates(champion, challenger):
    """Predeclared tolerances; no final-test values are inputs."""
    c, n = champion, challenger
    checks = {
        "rmse_within_2pct": n["rmse"] <= c["rmse"] * 1.02,
        "mae_within_2pct": n["mae"] <= c["mae"] * 1.02,
        "recall_floor": n["near_failure_recall"] is not None
        and c["near_failure_recall"] is not None
        and n["near_failure_recall"] >= max(0.90, c["near_failure_recall"] - 0.02),
        "false_alerts": n["false_alerts"] <= c["false_alerts"] + 1,
        "missed_critical": n["missed_critical"] <= c["missed_critical"],
        "finite_metrics": all(np.isfinite(n[k]) for k in ["mae", "rmse", "nasa_score"]),
    }
    return {
        "eligible": bool(all(checks.values())),
        "checks": {k: bool(v) for k, v in checks.items()},
        "policy": "v1: RMSE/MAE <=102%; recall >=max(90%, champion-2pp); false alerts <=+1; misses <=champion",
    }


def retrain(root):
    root = Path(root)
    record = active(root)
    champion = load_model(artifact(root, record))
    meta = record["metadata"]
    train_path = root / "data/raw/train_FD001.txt"
    if digest(train_path) != meta["training_sha256"]:
        raise ValueError("Training source changed; establish a new validated split before retraining")
    train = load(train_path)  # Never reads test_FD001 or RUL_FD001.
    y = targets(train)
    fit = train[train.engine_id.isin(meta["fit_engines"])]
    val = train[train.engine_id.isin(meta["validation_engines"])]
    if set(fit.engine_id) & set(val.engine_id) or len(fit) + len(val) != len(train):
        raise ValueError("Invalid engine-separated split")
    ids = endpoints(val)
    x = engineer(fit, champion.columns, champion.window)
    xv = engineer(val, champion.columns, champion.window).loc[ids]
    target = y.loc[fit.index].clip(upper=meta["cap"]) if meta.get("cap") else y.loc[fit.index]
    # Production estimator has seen all training engines. Reconstruct both recipes on fitting engines.
    baseline = clone(champion.model).fit(x, target)
    challenger = clone(champion.model)
    if not {"n_estimators", "random_state"}.issubset(challenger.get_params()):
        raise ValueError("This retraining recipe requires a tree ensemble champion")
    challenger.set_params(n_estimators=160, random_state=43)
    challenger.fit(x, target)
    outcomes = {}
    pred_frame = val.loc[ids, ["engine_id", "cycle"]].copy()
    pred_frame["actual_rul"] = y.loc[ids]
    for name, estimator in [("champion", baseline), ("challenger", challenger)]:
        pred = np.maximum(0, estimator.predict(xv))
        outcomes[name] = {**metrics(y.loc[ids], pred), **operational(y.loc[ids], pred)}
        pred_frame[name] = pred
    outcome = {
        "champion_version": record["version"],
        **outcomes,
        **gates(**outcomes),
        "validation_engines": meta["validation_engines"],
        "method": "both recipes refit on original 75 fitting engines",
    }
    folder = root / "artifacts/mlops/retraining" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    write_json(folder / "evaluation.json", outcome)
    pred_frame.to_csv(folder / "validation_predictions.csv", index=False)
    model_meta = {
        **meta,
        "source_run_id": uuid.uuid4().hex,
        "run_id": uuid.uuid4().hex,
        "created_at": now(),
        "candidate": "extra_trees_160_seed43",
        "seed": 43,
        "imported_benchmark": False,
        "benchmark": None,
        "validation": outcomes["challenger"],
    }
    # Refit deployment candidate after evaluation, still no automatic promotion.
    challenger.fit(
        engineer(train, champion.columns, champion.window),
        y.clip(upper=meta["cap"]) if meta.get("cap") else y,
    )
    bundle = ModelBundle(challenger, champion.columns, champion.window, model_meta)
    joblib.dump(bundle, folder / "model.joblib")
    run_id = log_run(
        root,
        "challenger-extra-trees-160-seed43",
        {
            "model_type": type(challenger).__name__,
            "parameters": challenger.get_params(),
            "feature_version": FEATURE_VERSION,
            "window": bundle.window,
            "channels": bundle.columns,
            "target_cap": meta.get("cap"),
            "seed": 43,
            "training_sha256": meta["training_sha256"],
            "training_time": model_meta["created_at"],
        },
        {f"validation_{k}": v for k, v in outcomes["challenger"].items()},
        [folder / "evaluation.json", folder / "validation_predictions.csv"],
    )
    version = register(root, folder / "model.joblib", run_id, model_meta)
    outcome["version"] = version
    write_json(folder / "evaluation.json", outcome)
    with connection(root) as db:
        db.execute(
            "INSERT INTO challenger_evaluations VALUES (?,?,?,?,?)",
            (version, record["version"], now(), int(outcome["eligible"]), json.dumps(outcome)),
        )
    write_json(root / "artifacts/mlops/latest_challenger.json", outcome)
    return outcome


def qualify(root, version):
    record = get_version(root, version)
    artifact(root, record)
    evidence = Path(root) / "artifacts/mlops" / f"tests_v{version}.xml"
    before = source_hash(root)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={evidence}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    evidence.with_suffix(".log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0 or before != source_hash(root):
        raise ValueError(f"Tests failed or source changed. Inspect {evidence.with_suffix('.log')}")
    suites = ET.parse(evidence).getroot()
    tests = sum(int(s.get("tests", 0)) for s in suites.iter("testsuite"))
    if tests == 0:
        raise ValueError("No tests collected")
    with connection(root) as db:
        db.execute(
            "INSERT OR REPLACE INTO test_attestations VALUES (?,?,?,?,?,?)",
            (version, now(), before, record["sha256"], tests, evidence.relative_to(root).as_posix()),
        )
    return {"version": version, "tests_passed": tests, "source_hash": before}
