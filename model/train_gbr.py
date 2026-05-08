"""
model/train_gbr.py
------------------
Training script for the Multi-Output GradientBoosting Regressor.

Wraps ``GradientBoostingRegressor`` in ``MultiOutputRegressor`` to
produce 5-step forecasts across 3 resource dimensions simultaneously.

Input:  Flattened rolling window of 20 steps × 3 resources = 60 features
Output: Next 5 steps × 3 resources = 15 values

Usage:
    python model/train_gbr.py

Outputs:
    model/saved_models/gbr_model.pkl   (MultiOutputRegressor)
    model/saved_models/scaler.pkl      (MinMaxScaler fitted on 3 features)
    model/saved_models/gbr_meta.json   (architecture metadata)
"""

import os
import sys
import json
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure the root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.workload_generator import generate_multivariate

WINDOW_SIZE      = 20
FORECAST_HORIZON = 5
N_FEATURES       = 3
MODEL_DIR        = os.path.join(os.path.dirname(__file__), "saved_models")


def build_dataset(data: np.ndarray, window: int, horizon: int):
    """
    Build sliding-window dataset from multivariate time series.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_steps, n_features).
    window : int
        Number of past steps used as input features.
    horizon : int
        Number of future steps to predict.

    Returns
    -------
    X : np.ndarray, shape (n_samples, window * n_features)
    y : np.ndarray, shape (n_samples, horizon * n_features)
    """
    n_features = data.shape[1]
    X, y = [], []
    for i in range(len(data) - window - horizon + 1):
        # Flatten the window: (window, n_features) → (window * n_features,)
        X.append(data[i: i + window].flatten())
        # Flatten the target horizon: (horizon, n_features) → (horizon * n_features,)
        y.append(data[i + window: i + window + horizon].flatten())
    return np.array(X), np.array(y)


def train(data: np.ndarray | None = None, verbose: bool = True) -> dict:
    """
    Train the multi-output GBR and report per-resource metrics.

    Parameters
    ----------
    data : np.ndarray, optional
        Shape (n_steps, 3). If None, generates diverse workload data.

    Returns
    -------
    dict
        Per-resource and overall R², RMSE, MAE.
    """
    if data is None:
        if verbose:
            print("Generating multi-resource training data for GBR...")
        segments = [
            generate_multivariate("gradual",  steps=500, seed=1),
            generate_multivariate("spike",    steps=500, seed=2),
            generate_multivariate("periodic", steps=500, seed=3),
            generate_multivariate("combined", steps=600, seed=4),
            generate_multivariate("gradual",  steps=400, seed=5),
            generate_multivariate("spike",    steps=400, seed=6),
            generate_multivariate("periodic", steps=400, seed=7),
        ]
        data = np.vstack(segments)

    if verbose:
        print(f"  Training data: {data.shape[0]} steps × {data.shape[1]} features")

    # Fit scaler on the raw features (3 columns)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X_all, y_all = build_dataset(data_scaled, WINDOW_SIZE, FORECAST_HORIZON)

    if verbose:
        print(f"  Dataset: {X_all.shape[0]} samples, {X_all.shape[1]} input features, "
              f"{y_all.shape[1]} output values")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    if verbose:
        print("  Training MultiOutputRegressor(GBR)...")

    base_estimator = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model = MultiOutputRegressor(base_estimator, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Reshape predictions and targets: (n_samples, horizon * n_features) → per-resource
    n_test = len(y_test)
    y_pred_r = y_pred.reshape(n_test, FORECAST_HORIZON, N_FEATURES)
    y_test_r = y_test.reshape(n_test, FORECAST_HORIZON, N_FEATURES)

    # Inverse-scale for reporting
    resource_names = ["cpu", "memory", "network"]
    metrics = {}
    all_pred, all_true = [], []

    for f_idx in range(N_FEATURES):
        pred_col = y_pred_r[:, :, f_idx].flatten()
        true_col = y_test_r[:, :, f_idx].flatten()

        # Inverse scale
        pred_full = np.zeros((len(pred_col), N_FEATURES))
        true_full = np.zeros((len(true_col), N_FEATURES))
        pred_full[:, f_idx] = pred_col
        true_full[:, f_idx] = true_col
        pred_inv = scaler.inverse_transform(pred_full)[:, f_idx]
        true_inv = scaler.inverse_transform(true_full)[:, f_idx]

        rmse = float(np.sqrt(mean_squared_error(true_inv, pred_inv)))
        mae  = float(mean_absolute_error(true_inv, pred_inv))
        r2   = float(r2_score(true_inv, pred_inv))

        name = resource_names[f_idx]
        metrics[f"{name}_rmse"] = rmse
        metrics[f"{name}_mae"]  = mae
        metrics[f"{name}_r2"]   = r2

        all_pred.extend(pred_inv.tolist())
        all_true.extend(true_inv.tolist())

    # Overall metrics
    metrics["rmse"] = float(np.sqrt(mean_squared_error(all_true, all_pred)))
    metrics["mae"]  = float(mean_absolute_error(all_true, all_pred))
    metrics["r2"]   = float(r2_score(all_true, all_pred))

    if verbose:
        print(f"\n  GBR Test Results:")
        for name in resource_names:
            print(f"    {name.upper()} — R²={metrics[f'{name}_r2']:.4f}  "
                  f"RMSE={metrics[f'{name}_rmse']:.4f}  MAE={metrics[f'{name}_mae']:.4f}")
        print(f"    Overall — R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}")

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  os.path.join(MODEL_DIR, "gbr_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    gbr_meta = {
        "n_features":       N_FEATURES,
        "window_size":      WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
    }
    with open(os.path.join(MODEL_DIR, "gbr_meta.json"), "w") as f:
        json.dump(gbr_meta, f, indent=2)

    if verbose:
        print(f"  Model saved to {MODEL_DIR}/")

    return metrics


if __name__ == "__main__":
    metrics = train()
    print(f"\nGBR training complete. Overall R² = {metrics['r2']:.4f}")
