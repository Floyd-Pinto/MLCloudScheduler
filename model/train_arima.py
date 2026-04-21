"""
model/train_arima.py
---------------------
Baseline ARIMA model for workload forecasting (comparison baseline).

Usage:
    pip install statsmodels
    python model/train_arima.py

The ARIMA model is trained and evaluated inline (no artifact saved).
It is called from the backend's evaluation service for baseline comparison.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.workload_generator import generate_gradual, generate_periodic, generate_combined
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_arima(series: np.ndarray, train_ratio: float = 0.8,
                   order: tuple = (2, 1, 2)) -> dict:
    """
    Fit ARIMA on the train split, walk-forward forecast on test split.
    Returns RMSE, MAE, R².
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        warnings.filterwarnings("ignore")
    except ImportError:
        print("statsmodels not installed. Run: pip install statsmodels")
        return {}

    n_train = int(len(series) * train_ratio)
    train, test = series[:n_train], series[n_train:]

    history = list(train)
    predictions = []

    for t in range(len(test)):
        try:
            model = ARIMA(history, order=order)
            fit   = model.fit()
            yhat  = fit.forecast(steps=1)[0]
        except Exception:
            yhat = history[-1]  # fallback: last value
        predictions.append(yhat)
        history.append(test[t])

    predictions = np.array(predictions)
    rmse = float(np.sqrt(mean_squared_error(test, predictions)))
    mae  = float(mean_absolute_error(test, predictions))
    r2   = float(r2_score(test, predictions))

    return {"rmse": rmse, "mae": mae, "r2": r2, "n_test": len(test)}


if __name__ == "__main__":
    print("Evaluating ARIMA baseline on multiple workload patterns...\n")
    patterns = {
        "gradual":  generate_gradual(steps=200, seed=42),
        "periodic": generate_periodic(steps=200, seed=42),
        "combined": generate_combined(steps=300, seed=42),
    }
    for name, series in patterns.items():
        result = evaluate_arima(series)
        if result:
            print(f"[{name:10s}]  RMSE={result['rmse']:.3f}  MAE={result['mae']:.3f}  R²={result['r2']:.3f}")
