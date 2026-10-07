import os
import json
import logging
import uuid
from time import perf_counter
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, create_model
from starlette.concurrency import run_in_threadpool

from maintenance.data import COLUMNS
from maintenance.model import load_model
from maintenance.mlops.registry import active, artifact
from maintenance.mlops.store import root_path
from maintenance.mlops.telemetry import record_event

# Explicit generated schema exposes all 26 named fields in OpenAPI.
Observation = create_model(
    "Observation",
    __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
    **{c: (int, Field(gt=0, strict=True)) if c in ["engine_id", "cycle"] else (float, ...) for c in COLUMNS},
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[Observation] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=4)
def _load_version(path, checksum, version, run_id, feature_version, candidate, benchmark):
    model = load_model(Path(path))
    model.metadata = {
        **model.metadata,
        "registry_version": version,
        "mlflow_run_id": run_id,
        "feature_version": feature_version,
        "model_type": candidate,
        "benchmark": json.loads(benchmark),
    }
    return model


def get_model():
    # Explicit legacy/offline override remains supported; normal serving always resolves the registry.
    if os.environ.get("MAINTENANCE_MODEL"):
        return load_model(Path(os.environ["MAINTENANCE_MODEL"]))
    record = active(root_path())
    path = artifact(root_path(), record)
    return _load_version(
        str(path),
        record["sha256"],
        record["version"],
        record["mlflow_run_id"],
        record["metadata"]["feature_version"],
        record["metadata"].get("candidate"),
        json.dumps(record["metadata"].get("benchmark")),
    )


get_model.cache_clear = _load_version.cache_clear


app = FastAPI(
    title="FD001 Maintenance Decision Support",
    version="1.0.0",
    description="Simulated-engine RUL estimates. Project thresholds; not safety-certified.",
)


@app.middleware("http")
async def inference_telemetry(request: Request, call_next):
    if request.url.path != "/predict" or request.method != "POST":
        return await call_next(request)
    started = perf_counter()
    version, code = None, 500
    request.state.prediction = None
    try:
        try:
            model = await run_in_threadpool(get_model)
            request.state.model = model
            version = model.metadata.get("registry_version", model.metadata["run_id"])
        except (FileNotFoundError, ValueError, OSError):
            request.state.model = None
        response = await call_next(request)
        code = response.status_code
        return response
    finally:
        try:
            await run_in_threadpool(
                record_event,
                root_path(),
                uuid.uuid4().hex,
                version,
                code,
                (perf_counter() - started) * 1000,
                request.state.prediction,
                "request_rejected" if code >= 400 else None,
            )
        except Exception:
            logging.exception("Failed to persist inference telemetry")


@app.get("/health")
def health():
    try:
        model = get_model()
        return {
            "status": "ready",
            "model_version": model.metadata.get("registry_version", model.metadata["run_id"]),
        }
    except (FileNotFoundError, ValueError, OSError):
        raise HTTPException(503, "Model unavailable; run python mlops.py bootstrap")


@app.get("/model-info")
def model_info():
    health()
    model = get_model()
    return {
        "metadata": {
            k: model.metadata.get(k)
            for k in [
                "registry_version",
                "mlflow_run_id",
                "run_id",
                "model_type",
                "feature_version",
                "benchmark",
            ]
        },
        "active_model_version": model.metadata.get("registry_version", model.metadata["run_id"]),
        "required_recent_cycles": model.window,
        "thresholds": {"Critical": 20, "Schedule Maintenance": 40, "Monitor": 80},
        "input": "One engine; consecutive observations ending at the desired prediction cycle",
    }


@app.post("/predict")
def predict(request: PredictionRequest, context: Request):
    model = getattr(context.state, "model", None)
    if model is None:
        raise HTTPException(503, "No usable production model; run python mlops.py bootstrap")
    try:
        frame = pd.DataFrame([r.model_dump() for r in request.observations], columns=COLUMNS)
        prediction = model.latest(frame)
        prediction.update(
            model_version=model.metadata.get("registry_version", model.metadata["run_id"]),
            source_run_id=model.metadata["run_id"],
            mlflow_run_id=model.metadata.get("mlflow_run_id"),
            feature_version=model.metadata.get("feature_version"),
        )
        context.state.prediction = prediction
        return prediction
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
