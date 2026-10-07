"""Causal, engine-local trailing features, independent of target labels."""

import pandas as pd


def informative(frame: pd.DataFrame) -> list[str]:
    # A tiny relative range excludes effectively constant channels on fitting engines only.
    cols = [c for c in frame if c.startswith(("sensor_", "setting_"))]
    return [
        c
        for c in cols
        if frame[c].nunique() > 2 and (frame[c].max() - frame[c].min()) / max(abs(frame[c].mean()), 1) > 1e-5
    ]


def engineer(frame: pd.DataFrame, columns: list[str], window: int) -> pd.DataFrame:
    frame = frame.sort_values(["engine_id", "cycle"])
    result = frame[["cycle"] + columns].copy()
    for col in columns:
        group = frame.groupby("engine_id", sort=False)[col]
        result[f"{col}_mean"] = group.transform(lambda x: x.rolling(window, min_periods=1).mean())
        result[f"{col}_std"] = group.transform(lambda x: x.rolling(window, min_periods=1).std(ddof=0))
        result[f"{col}_delta"] = group.diff().fillna(0)
    return result
