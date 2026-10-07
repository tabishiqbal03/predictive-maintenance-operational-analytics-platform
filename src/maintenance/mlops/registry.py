import json
from pathlib import Path
import shutil
import uuid

from maintenance.model import load_model
from maintenance.mlops.store import connection, digest, initialize, now, rows, source_hash
from maintenance.mlops.tracking import register_mlflow


def register(root, path, run_id, metadata):
    initialize(root)
    load_model(Path(path))  # Validate trusted serialized bundle before registration.
    folder = Path(root) / "artifacts/mlops/models" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    target = folder / "model.joblib"
    shutil.copy2(path, target)
    version = register_mlflow(root, run_id, target)
    with connection(root) as db:
        db.execute(
            "INSERT INTO model_versions VALUES (?,?,?,?,?,NULL,'candidate',?)",
            (
                version,
                run_id,
                target.relative_to(root).as_posix(),
                digest(target),
                now(),
                json.dumps(metadata),
            ),
        )
    return version


def get_version(root, version):
    initialize(root)
    result = rows(root, "SELECT * FROM model_versions WHERE version=?", (str(version),))
    if not result:
        raise ValueError(f"Unknown model version {version}")
    result[0]["metadata"] = json.loads(result[0]["metadata"])
    return result[0]


def active(root):
    initialize(root)
    result = rows(root, "SELECT version FROM production_model WHERE singleton=1")
    if not result:
        raise FileNotFoundError("No production model. Run python mlops.py bootstrap first.")
    return get_version(root, result[0]["version"])


def artifact(root, record):
    path = (Path(root) / record["model_path"]).resolve()
    if not path.is_relative_to(Path(root).resolve() / "artifacts/mlops/models"):
        raise ValueError("Registry artifact path is outside the model store")
    if not path.exists() or digest(path) != record["sha256"]:
        raise ValueError("Model artifact missing or checksum mismatch")
    return path


def switch(root, version, reason, rollback=False, bootstrap=False):
    if not reason.strip():
        raise ValueError("An audit reason is required")
    record = get_version(root, version)
    artifact(root, record)
    # Acquire write lock before validating current champion and updating all lifecycle state.
    with connection(root) as db:
        db.execute("BEGIN IMMEDIATE")
        current = db.execute("SELECT version FROM production_model WHERE singleton=1").fetchone()
        previous = current[0] if current else None
        if previous == str(version):
            return
        if bootstrap:
            if previous is not None:
                raise ValueError("Bootstrap cannot overwrite production")
            if not record["metadata"].get("imported_benchmark"):
                raise ValueError("Bootstrap requires an imported verified benchmark")
        elif rollback:
            if record["approved_at"] is None:
                raise ValueError("Rollback target was never approved for production")
        else:
            evaluation = db.execute(
                "SELECT * FROM challenger_evaluations WHERE version=?", (version,)
            ).fetchone()
            if not evaluation or not evaluation["eligible"] or evaluation["champion_version"] != previous:
                raise ValueError("Gates failed or evaluation is stale against the current champion")
            tests = db.execute("SELECT * FROM test_attestations WHERE version=?", (version,)).fetchone()
            if (
                not tests
                or tests["source_hash"] != source_hash(root)
                or tests["model_hash"] != record["sha256"]
            ):
                raise ValueError("Passing test evidence missing or stale; run qualify")
        stamp = now()
        if previous:
            db.execute("UPDATE model_versions SET lifecycle='previous' WHERE version=?", (previous,))
        db.execute(
            "UPDATE model_versions SET lifecycle='production', approved_at=COALESCE(approved_at,?) WHERE version=?",
            (stamp, version),
        )
        db.execute("INSERT OR REPLACE INTO production_model VALUES (1,?,?)", (version, stamp))
        db.execute(
            "INSERT INTO lifecycle_events(timestamp,action,previous_version,version,reason) VALUES (?,?,?,?,?)",
            (
                stamp,
                "bootstrap" if bootstrap else "rollback" if rollback else "promotion",
                previous,
                version,
                reason,
            ),
        )
