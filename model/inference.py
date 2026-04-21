"""
model/inference.py  (updated — supports GBR, LSTM, ARIMA, Combined)
"""

import os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


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

    raise ValueError(f"Unknown model_type '{model_type}'. Supported: gbr, lstm, arima, combined")


def predict(history: list[float], model_type: str = "gbr", window_size: int = 10) -> float:
    """Unified prediction entry point."""
    if model_type == "gbr":
        obj = _get_model("gbr")
        window = np.array(history[-window_size:]).reshape(1, -1)
        window_s = obj["scaler"].transform(window)
        return float(obj["model"].predict(window_s)[0])

    if model_type in ("lstm", "arima", "combined"):
        m = _get_model(model_type)
        return m.predict(history)

    raise ValueError(f"Unknown model_type: {model_type}")


def predict_all(history: list[float]) -> dict:
    """Run all available models and return dict of predictions."""
    results = {}
    for mt in ("gbr", "lstm", "arima", "combined"):
        try:
            results[mt] = round(predict(history, model_type=mt), 4)
        except Exception as e:
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
