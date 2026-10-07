import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics(actual, predicted) -> dict:
    error = np.asarray(predicted) - np.asarray(actual)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "nasa_score": float(np.expm1(np.where(error < 0, -error / 13, error / 10)).sum()),
        "n": len(error),
    }


def operational(actual, predicted, threshold: int = 40) -> dict:
    """Vary the prediction trigger only; false-alert truth stays at actual RUL >40."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    critical, alert = actual <= 20, predicted <= threshold
    false_alert = alert & (actual > 40)
    return {
        "alert_threshold": threshold,
        "near_failure_count": int(critical.sum()),
        "near_failure_recall": float(alert[critical].mean()) if critical.any() else None,
        "missed_critical": int((critical & ~alert).sum()),
        "false_alerts": int(false_alert.sum()),
        "alerts": int(alert.sum()),
        "false_alert_fraction": float(false_alert.sum() / alert.sum()) if alert.any() else None,
    }
