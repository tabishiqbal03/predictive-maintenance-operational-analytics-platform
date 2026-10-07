"""Explicit MLflow client calls avoid process-global tracking state."""

from pathlib import Path

from mlflow import MlflowClient

NAME = "FD001-RUL"


def client(root):
    directory = Path(root).resolve() / "artifacts/mlops"
    directory.mkdir(parents=True, exist_ok=True)
    return MlflowClient(tracking_uri="sqlite:///" + (directory / "mlflow.sqlite").as_posix())


def log_run(root, name, params, metrics, artifacts=(), tags=None):
    api = client(root)
    experiment = api.get_experiment_by_name(NAME)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else api.create_experiment(
            NAME, artifact_location=(Path(root).resolve() / "artifacts/mlops/mlruns").as_uri()
        )
    )
    run = api.create_run(
        experiment_id, tags={"mlflow.runName": name, "dataset": "NASA simulated FD001", **(tags or {})}
    )
    run_id = run.info.run_id
    try:
        for key, value in params.items():
            api.log_param(run_id, key, str(value)[:6000])
        for key, value in metrics.items():
            if isinstance(value, (float, int)) and value is not None:
                api.log_metric(run_id, key, value)
        for path in artifacts:
            path = Path(path)
            if path.is_file():
                api.log_artifact(run_id, str(path), "evidence")
        api.set_terminated(run_id, "FINISHED")
    except Exception:
        api.set_terminated(run_id, "FAILED")
        raise
    return run_id


def register_mlflow(root, run_id, model_path):
    api = client(root)
    try:
        api.get_registered_model(NAME)
    except Exception as exc:
        if getattr(exc, "error_code", None) != "RESOURCE_DOES_NOT_EXIST":
            raise
        api.create_registered_model(
            NAME, description="Trusted native ModelBundle; deployment via local registry"
        )
    api.log_artifact(run_id, str(model_path), "bundle")
    uri = api.get_run(run_id).info.artifact_uri + "/bundle"
    return str(api.create_model_version(NAME, uri, run_id=run_id).version)
