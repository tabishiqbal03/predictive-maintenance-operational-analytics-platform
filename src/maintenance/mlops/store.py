import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).isoformat()


def root_path():
    return Path(os.environ.get("MAINTENANCE_ROOT", Path(__file__).resolve().parents[3])).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


@contextmanager
def connection(root):
    folder = Path(root) / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(folder / "maintenance.sqlite", timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize(root):
    with connection(root) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS model_versions (
          version TEXT PRIMARY KEY, mlflow_run_id TEXT NOT NULL, model_path TEXT NOT NULL,
          sha256 TEXT NOT NULL, created_at TEXT NOT NULL, approved_at TEXT,
          lifecycle TEXT NOT NULL, metadata TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS production_model (
          singleton INTEGER PRIMARY KEY CHECK(singleton=1),
          version TEXT NOT NULL REFERENCES model_versions(version), updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS lifecycle_events (
          id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, action TEXT NOT NULL,
          previous_version TEXT, version TEXT NOT NULL, reason TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS challenger_evaluations (
          version TEXT PRIMARY KEY REFERENCES model_versions(version), champion_version TEXT NOT NULL,
          timestamp TEXT NOT NULL, eligible INTEGER NOT NULL, evidence TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS test_attestations (
          version TEXT PRIMARY KEY REFERENCES model_versions(version), timestamp TEXT NOT NULL,
          source_hash TEXT NOT NULL, model_hash TEXT NOT NULL, tests INTEGER NOT NULL, evidence_path TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS inference_events (
          id INTEGER PRIMARY KEY, request_id TEXT NOT NULL, timestamp TEXT NOT NULL,
          model_version TEXT, status_code INTEGER NOT NULL, latency_ms REAL NOT NULL,
          predicted_rul REAL, maintenance_status TEXT, error_type TEXT);
        CREATE TABLE IF NOT EXISTS monitoring_runs (
          id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, model_version TEXT NOT NULL,
          scenario TEXT NOT NULL, status TEXT NOT NULL, report_path TEXT NOT NULL, summary TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS inference_time ON inference_events(timestamp);
        CREATE VIEW IF NOT EXISTS service_summary AS
          SELECT model_version, COUNT(*) AS requests,
          SUM(CASE WHEN status_code>=400 THEN 1 ELSE 0 END) AS errors,
          AVG(latency_ms) AS mean_latency_ms, MAX(latency_ms) AS max_latency_ms
          FROM inference_events GROUP BY model_version;
        """)


def rows(root, query, params=()):
    with connection(root) as db:
        return [dict(row) for row in db.execute(query, params).fetchall()]


def source_hash(root):
    """Bind test evidence to code, tests and locked environment, not changing result files."""
    root = Path(root)
    files = [root / "pyproject.toml", root / "uv.lock", root / "run_pipeline.py", root / "mlops.py"]
    for name in ["src/maintenance", "api", "app", "tests", "scripts"]:
        files.extend((root / name).rglob("*.py"))
    hasher = hashlib.sha256()
    for path in sorted(set(files)):
        if path.exists():
            hasher.update(path.relative_to(root).as_posix().encode())
            hasher.update(path.read_bytes())
    return hasher.hexdigest()
