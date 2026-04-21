"""
model/train_gbr.py
------------------
Training script for the GradientBoostingRegressor (primary ML model).

Usage:
    python model/train_gbr.py

Outputs:
    model/saved_models/gbr_model.pkl
    model/saved_models/scaler.pkl
"""

import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure the root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.workload_generator import generate_gradual, generate_spike, generate_periodic, generate_combined

WINDOW_SIZE = 10
HORIZON     = 5
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "saved_models")


def build_dataset(series: np.ndarray, window: int, horizon: int):
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i: i + window])
        y.append(series[i + window + horizon - 1])
    return np.array(X), np.array(y)


def train():
    print("Generating training data...")
    patterns = [
        generate_gradual(steps=500, seed=1),
        generate_spike(steps=500, seed=2),
        generate_periodic(steps=500, seed=3),
        generate_combined(steps=600, seed=4),
        generate_gradual(steps=400, seed=5),
        generate_spike(steps=400, spike_at=200, seed=6),
        generate_periodic(steps=400, period=30, seed=7),
    ]

    X_all, y_all = [], []
    for series in patterns:
        X, y = build_dataset(series, WINDOW_SIZE, HORIZON)
        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    print(f"Dataset: {X_all.shape[0]} samples, {X_all.shape[1]} features each")

    scaler  = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_all)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_all, test_size=0.2, random_state=42
    )

    print("Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse  = mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\nTest Results:")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  os.path.join(MODEL_DIR, "gbr_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"\nModel saved to {MODEL_DIR}/")

    return {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    metrics = train()
    print(f"\nTraining complete. R² = {metrics['r2']:.4f}")
