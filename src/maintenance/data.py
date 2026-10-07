"""Strict input validation; no silent imputation or record repair."""

import hashlib
import io
import json
import logging
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
from zipfile import ZipFile

import numpy as np
import pandas as pd

COLUMNS = (
    ["engine_id", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
)
URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]


def download(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if all((folder / name).exists() for name in FILES):
        logging.info("Using existing FD001 files")
        return
    logging.info("Downloading NASA-linked archive: %s", URL)
    try:
        with urlopen(URL, timeout=120) as response:
            payload = response.read()
    except (URLError, OSError) as exc:
        raise ValueError(
            f"Download failed: {exc}. Download the NASA-linked archive manually and place "
            f"{', '.join(FILES)} in {folder}."
        ) from exc
    found = {}

    def scan(blob: bytes) -> None:
        with ZipFile(io.BytesIO(blob)) as archive:
            for member in archive.namelist():
                name = Path(member).name
                if name in FILES:
                    found[name] = archive.read(member)
                elif name.lower().endswith(".zip"):
                    scan(archive.read(member))

    scan(payload)
    if set(found) != set(FILES):
        raise ValueError(f"Archive missing FD001 files: {set(FILES) - set(found)}")
    for name, blob in found.items():
        (folder / name).write_bytes(blob)
    (folder / "provenance.json").write_text(
        json.dumps(
            {"url": URL, "sha256": {name: hashlib.sha256(blob).hexdigest() for name, blob in found.items()}},
            indent=2,
        ),
        encoding="utf-8",
    )


def validate(frame: pd.DataFrame, require_start: bool = True) -> pd.DataFrame:
    if list(frame.columns) != COLUMNS or frame.empty:
        raise ValueError("Expected nonempty 26-column C-MAPSS schema")
    if not all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes):
        raise ValueError("All input columns must be numeric")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Missing or non-finite values are not permitted")
    for col in ["engine_id", "cycle"]:
        if ((frame[col] < 1) | (frame[col] % 1 != 0)).any():
            raise ValueError(f"{col} must contain positive integers")
    if frame.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle records")
    result = frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True).copy()
    result[["engine_id", "cycle"]] = result[["engine_id", "cycle"]].astype(int)
    if result.groupby("engine_id").cycle.diff().dropna().ne(1).any():
        raise ValueError("Cycles must be consecutive within each engine")
    if require_start and result.groupby("engine_id").cycle.min().ne(1).any():
        raise ValueError("Dataset histories must start at cycle 1")
    return result


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python run_pipeline.py --download or add FD001 files to data/raw."
        )
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != 26:
        raise ValueError(f"Expected 26 columns, found {frame.shape[1]}")
    frame.columns = COLUMNS
    return validate(frame)


def targets(frame: pd.DataFrame) -> pd.Series:
    return (frame.groupby("engine_id").cycle.transform("max") - frame.cycle).rename("actual_rul")
