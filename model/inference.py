"""
model/inference.py
-------------------
Unified multi-resource prediction interface for all four models:
LSTM, ARIMA, Combined ensemble, and GBR.

All models accept a window of shape (20, 3) and return per-resource
5-step forecasts via a single ``predict()`` entry point.
"""

import os
import json
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

WINDOW_SIZE      = 20
FORECAST_HORIZON = 5
N_FEATURES       = 3


# ── per-model singletons ──────────────────────────────────────────────────────
_models: dict = {}


def _get_model(model_type: str):
    if model_type in _models:
        return _models[model_type]

    if model_type == "gbr":
        import joblib
        mp = os.path.join(MODEL_DIR, "gbr_model.pkl")
        sp = os.path.join(MODEL_DIR, "scaler.pkl")
        if not (os.path.exists(mp) and os.path.exists(sp)):
            raise FileNotFoundError("GBR model not found. Run model/train_gbr.py")
        obj = {"model": joblib.load(mp), "scaler": joblib.load(sp)}
        meta_path = os.path.join(MODEL_DIR, "gbr_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                obj["meta"] = json.load(f)
        _models["gbr"] = obj
        return obj

    if model_type == "lstm":
        from model.lstm_model import LSTMForecaster
        m = LSTMForecaster()
        if not m.load(MODEL_DIR):
            raise FileNotFoundError("LSTM model not found. Run model/train_all.py")
        _models["lstm"] = m
        return m

    if model_type == "arima":
        from model.arima_model import ARIMAForecaster
        m = ARIMAForecaster()
        if not m.load(MODEL_DIR):
            raise FileNotFoundError("ARIMA model not found. Run model/train_all.py")
        _models["arima"] = m
        return m

    if model_type == "combined":
        from model.combined_model import CombinedForecaster
        m = CombinedForecaster()
        if not m.load(MODEL_DIR):
            raise FileNotFoundError("Combined model not found. Run model/train_all.py")
        _models["combined"] = m
        return m

    raise ValueError(f"Unknown model_type '{model_type}'. "
                     f"Supported: gbr, lstm, arima, combined")


def predict(model_type: str, window: np.ndarray) -> dict:
    """
    Unified prediction interface for all models.

    Parameters
    ----------
    model_type : str
        One of "lstm", "arima", "combined", "gbr".
    window : np.ndarray
        Shape (20, 3) — last 20 steps of [cpu, memory, network_io].
        Also accepts shape (n,) for backward compatibility (single signal).

    Returns
    -------
    dict
        {
          "cpu":     [f1, f2, f3, f4, f5],
          "memory":  [f1, f2, f3, f4, f5],
          "network": [f1, f2, f3, f4, f5]
        }
    """
    # Normalize input to (window_size, n_features)
    if isinstance(window, list):
        window = np.array(window, dtype=np.float32)
    if window.ndim == 1:
        w = window[-WINDOW_SIZE:]
        window = np.column_stack([w, w * 0.7, w * 0.5])

    window = window[-WINDOW_SIZE:]  # ensure correct window size

    if model_type == "gbr":
        obj = _get_model("gbr")
        scaler = obj["scaler"]
        model = obj["model"]
        window_scaled = scaler.transform(window)
        X = window_scaled.flatten().reshape(1, -1)
        pred_flat = model.predict(X)[0]  # shape: (FORECAST_HORIZON * N_FEATURES,)
        pred = pred_flat.reshape(FORECAST_HORIZON, N_FEATURES)
        pred_inv = scaler.inverse_transform(pred)
        pred_inv = np.clip(pred_inv, 0, 100)
    elif model_type == "lstm":
        m = _get_model("lstm")
        pred_inv = m.predict(window)
        pred_inv = np.clip(pred_inv, 0, 100)
    elif model_type == "arima":
        m = _get_model("arima")
        pred_inv = m.predict(window, steps=FORECAST_HORIZON)
        pred_inv = np.clip(pred_inv, 0, 100)
    elif model_type == "combined":
        m = _get_model("combined")
        pred_inv = m.predict(window)
        pred_inv = np.clip(pred_inv, 0, 100)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return {
        "cpu":     [round(float(v), 4) for v in pred_inv[:, 0]],
        "memory":  [round(float(v), 4) for v in pred_inv[:, 1]],
        "network": [round(float(v), 4) for v in pred_inv[:, 2]],
    }


def predict_all(window: np.ndarray) -> dict:
    """Run all available models and return dict of per-model predictions."""
    results = {}
    for mt in ("gbr", "lstm", "arima", "combined"):
        try:
            results[mt] = predict(model_type=mt, window=window)
        except Exception:
            results[mt] = None
    return results


def model_is_ready(model_type: str = "gbr") -> bool:
    if model_type == "gbr":
        return (
            os.path.exists(os.path.join(MODEL_DIR, "gbr_model.pkl")) and
            os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl"))
        )
    if model_type == "lstm":
        return os.path.exists(os.path.join(MODEL_DIR, "lstm_model.pt"))
    if model_type == "arima":
        return os.path.exists(os.path.join(MODEL_DIR, "arima_meta.json"))
    if model_type == "combined":
        return os.path.exists(os.path.join(MODEL_DIR, "combined_meta.json"))
    return False


def all_model_statuses() -> dict:
    return {mt: model_is_ready(mt) for mt in ("gbr", "lstm", "arima", "combined")}


def invalidate_cache(model_type: str | None = None):
    """Clear singleton cache after retraining."""
    global _models
    if model_type:
        _models.pop(model_type, None)
    else:
        _models.clear()
