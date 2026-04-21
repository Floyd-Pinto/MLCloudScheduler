"""
model/combined_model.py
------------------------
Hybrid LSTM + ARIMA ensemble.
Weights are determined by inverse-RMSE so the better model gets more weight.
The combined prediction is:
    ŷ_combined = w_lstm * ŷ_lstm + w_arima * ŷ_arima
where w_lstm = (1/rmse_lstm) / (1/rmse_lstm + 1/rmse_arima)
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.lstm_model  import LSTMForecaster
from model.arima_model import ARIMAForecaster

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


class CombinedForecaster:
    """Weighted ensemble: LSTM + ARIMA."""

    def __init__(self):
        self.lstm      = LSTMForecaster()
        self.arima     = ARIMAForecaster()
        self.w_lstm    = 0.5
        self.w_arima   = 0.5
        self._ready    = False

    def fit(self, series: np.ndarray, verbose: bool = True) -> dict:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        if verbose:
            print("\n  [Combined] Training LSTM component...")
        lstm_metrics = self.lstm.fit(series, verbose=verbose)

        if verbose:
            print("\n  [Combined] Training ARIMA component...")
        arima_metrics = self.arima.fit(series, verbose=verbose)

        # Compute weights by inverse RMSE (lower RMSE → higher weight)
        r_l   = lstm_metrics["rmse"]  + 1e-6
        r_a   = arima_metrics["rmse"] + 1e-6
        inv_l = 1.0 / r_l
        inv_a = 1.0 / r_a
        total = inv_l + inv_a
        self.w_lstm  = inv_l / total
        self.w_arima = inv_a / total

        if verbose:
            print(f"\n  [Combined] w_lstm={self.w_lstm:.3f}  w_arima={self.w_arima:.3f}")

        # Evaluate combined on same test split (last 30%)
        test_size   = max(30, int(len(series) * 0.30))
        test_series = series[-test_size:]
        train_hist  = list(series[:-test_size])

        preds = []
        for t in range(len(test_series)):
            hist = train_hist + list(test_series[:t])
            try:
                p_lstm  = self.lstm.predict(hist)
            except Exception:
                p_lstm  = hist[-1]
            try:
                p_arima = self.arima.predict(hist, steps=5)
            except Exception:
                p_arima = hist[-1]
            preds.append(self.w_lstm * p_lstm + self.w_arima * p_arima)

        preds = np.array(preds)
        true  = test_series

        rmse = float(np.sqrt(mean_squared_error(true, preds)))
        mae  = float(mean_absolute_error(true, preds))
        r2   = float(r2_score(true, preds))

        self._ready = True
        return {
            "rmse": rmse, "mae": mae, "r2": r2,
            "lstm_rmse": lstm_metrics["rmse"], "arima_rmse": arima_metrics["rmse"],
            "w_lstm": round(self.w_lstm, 4), "w_arima": round(self.w_arima, 4),
        }

    def predict(self, history: list[float]) -> float:
        if not self._ready:
            raise RuntimeError("Combined model not trained.")
        try:
            p_lstm  = self.lstm.predict(history)
        except Exception:
            p_lstm  = history[-1]
        try:
            p_arima = self.arima.predict(history, steps=5)
        except Exception:
            p_arima = history[-1]
        return self.w_lstm * p_lstm + self.w_arima * p_arima

    def save(self, directory: str | None = None):
        d = directory or MODEL_DIR
        os.makedirs(d, exist_ok=True)
        self.lstm.save(d)
        self.arima.save(d)
        meta = {"w_lstm": self.w_lstm, "w_arima": self.w_arima}
        with open(os.path.join(d, "combined_meta.json"), "w") as f:
            json.dump(meta, f)

    def load(self, directory: str | None = None) -> bool:
        d = directory or MODEL_DIR
        path = os.path.join(d, "combined_meta.json")
        lstm_ok  = self.lstm.load(d)
        arima_ok = self.arima.load(d)
        if os.path.exists(path):
            with open(path) as f:
                meta = json.load(f)
            self.w_lstm  = meta.get("w_lstm",  0.5)
            self.w_arima = meta.get("w_arima", 0.5)
        self._ready = lstm_ok and arima_ok
        return self._ready

    def is_ready(self) -> bool:
        return self._ready


_combined_instance: CombinedForecaster | None = None

def get_combined() -> CombinedForecaster:
    global _combined_instance
    if _combined_instance is None:
        _combined_instance = CombinedForecaster()
        _combined_instance.load()
    return _combined_instance
