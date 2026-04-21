"""
model/train_lstm.py
-------------------
LSTM (deep learning) training script for workload forecasting.

Usage:
    pip install tensorflow
    python model/train_lstm.py

Outputs:
    model/saved_models/lstm_model.h5
    model/saved_models/lstm_scaler.pkl
"""

import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.workload_generator import generate_gradual, generate_spike, generate_periodic, generate_combined

WINDOW_SIZE = 20
HORIZON     = 5
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "saved_models")


def build_sequences(series: np.ndarray, window: int, horizon: int):
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i: i + window])
        y.append(series[i + window + horizon - 1])
    return np.array(X), np.array(y)


def train():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    except ImportError:
        print("TensorFlow not installed. Run: pip install tensorflow")
        return None

    print("Generating training data for LSTM...")
    patterns = [
        generate_gradual(steps=600, seed=10),
        generate_spike(steps=600, seed=11),
        generate_periodic(steps=600, seed=12),
        generate_combined(steps=700, seed=13),
        generate_gradual(steps=500, seed=14),
        generate_spike(steps=500, seed=15),
    ]

    X_all, y_all = [], []
    for series in patterns:
        # Scale each series independently then collect
        X, y = build_sequences(series, WINDOW_SIZE, HORIZON)
        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    # Scale
    scaler = MinMaxScaler()
    X_flat = X_all.reshape(-1, WINDOW_SIZE)
    X_scaled = scaler.fit_transform(X_flat).reshape(-1, WINDOW_SIZE, 1)
    y_min, y_max = y_all.min(), y_all.max()
    y_scaled = (y_all - y_min) / (y_max - y_min + 1e-8)

    split = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    print(f"Dataset: {X_all.shape[0]} sequences, window={WINDOW_SIZE}, horizon={HORIZON}")

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=80,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    # Evaluate (inverse transform)
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = y_pred_scaled * (y_max - y_min) + y_min
    y_true = y_test * (y_max - y_min) + y_min

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    print(f"\nLSTM Test Results:")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
    # Save scaler + y range for inverse transform
    meta = {"scaler": scaler, "y_min": y_min, "y_max": y_max}
    joblib.dump(meta, os.path.join(MODEL_DIR, "lstm_meta.pkl"))
    print(f"\nLSTM model saved to {MODEL_DIR}/")

    return {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    result = train()
    if result:
        print(f"\nLSTM training complete. R² = {result['r2']:.4f}")
