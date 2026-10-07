from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from maintenance.data import validate
from maintenance.decision import status
from maintenance.features import engineer


@dataclass
class ModelBundle:
    model: object
    columns: list[str]
    window: int
    metadata: dict

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.maximum(0, self.model.predict(engineer(frame, self.columns, self.window)))

    def latest(self, frame: pd.DataFrame) -> dict:
        frame = validate(frame, require_start=False)
        if frame.engine_id.nunique() != 1:
            raise ValueError("Supply exactly one engine per request")
        needed = min(self.window, int(frame.cycle.max()))
        if len(frame) < needed:
            raise ValueError(f"Supply at least {needed} consecutive recent cycles")
        value = float(self.predict(frame)[-1])
        return {
            "engine_id": int(frame.engine_id.iloc[-1]),
            "cycle": int(frame.cycle.iloc[-1]),
            "predicted_rul": value,
            "maintenance_status": status(value),
            "model_version": self.metadata["run_id"],
        }


def load_model(path: Path) -> ModelBundle:
    if not path.exists():
        raise FileNotFoundError("Model unavailable. Run python run_pipeline.py first.")
    # Only load trusted, locally generated artifacts; pickle is not an interchange format.
    return joblib.load(path)
