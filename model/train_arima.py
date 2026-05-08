"""
model/train_arima.py
---------------------
Training script for the multi-resource ARIMA forecaster.

Fits independent ARIMA(p,d,q) models per resource channel with
automatic order selection via AIC grid search. Walk-forward
validation on a 300-step segment.

Usage:
    python model/train_arima.py

Outputs:
    model/saved_models/arima_meta.json      (per-channel orders, n_features)
    model/saved_models/arima_history.pkl    (training history for prediction)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.workload_generator import generate_multivariate
from model.arima_model import ARIMAForecaster


def train(data: np.ndarray | None = None, verbose: bool = True) -> dict:
    """
    Train the multi-resource ARIMA model.

    Uses a representative 300-step segment for walk-forward validation
    to keep training time manageable.

    Parameters
    ----------
    data : np.ndarray, optional
        Shape (n_steps, 3). If None, generates default data.

    Returns
    -------
    dict
        Per-resource R², RMSE, MAE metrics.
    """
    if data is None:
        if verbose:
            print("Generating multi-resource training data for ARIMA...")
        data = generate_multivariate("combined", steps=300, seed=42)

    if verbose:
        print(f"  Training data: {data.shape[0]} steps × {data.shape[1]} features")

    model = ARIMAForecaster()
    metrics = model.fit(data, verbose=verbose)
    model.save()

    if verbose:
        print(f"\n  ARIMA Results:")
        for key in ["cpu_r2", "memory_r2", "network_r2", "r2"]:
            if key in metrics:
                label = key.replace("_", " ").title()
                print(f"    {label}: {metrics[key]:.4f}")

    return metrics


if __name__ == "__main__":
    print("Training multi-resource ARIMA on combined workload...\n")
    data = generate_multivariate("combined", steps=300, seed=42)
    result = train(data)
    if result:
        print(f"\nARIMA training complete. Overall R² = {result['r2']:.4f}")
