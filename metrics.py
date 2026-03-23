import math
from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    n, m = len(x), len(y)
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1],
            )

    return float(dtw[n, m])


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(pearsonr(y_true, y_pred)[0])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r": safe_pearsonr(y_true, y_pred),
        "dtw": dtw_distance(y_true, y_pred),
    }


def summarize_cv_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    metric_names = fold_metrics[0].keys()

    for name in metric_names:
        values = np.array([m[name] for m in fold_metrics], dtype=np.float64)
        valid = values[~np.isnan(values)]
        mean_value = float(np.mean(valid))
        se_value = float(np.std(valid, ddof=1) / math.sqrt(len(valid))) if len(valid) > 1 else 0.0
        summary[name] = {
            "mean": mean_value,
            "se": se_value,
        }

    return summary