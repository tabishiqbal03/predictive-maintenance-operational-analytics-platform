import numpy as np


def status(rul: float) -> str:
    if not np.isfinite(rul) or rul < 0:
        raise ValueError("RUL must be finite and nonnegative")
    if rul <= 20:
        return "Critical"
    if rul <= 40:
        return "Schedule Maintenance"
    if rul <= 80:
        return "Monitor"
    return "Healthy"
