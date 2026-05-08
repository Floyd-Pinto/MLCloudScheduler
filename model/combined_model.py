"""
model/combined_model.py
------------------------
Per-Resource Adaptive Hybrid Ensemble: LSTM + ARIMA.

Computes per-resource weights using inverse-RMSE from the individual
model training metrics, enabling resource-specific model preference.
For example, LSTM may dominate for CPU prediction while ARIMA may be
preferred for memory forecasting.

Weighting scheme:
  For each resource r ∈ {cpu, memory, network}:
    w_lstm_r  = (1/rmse_lstm_r) / (1/rmse_lstm_r + 1/rmse_arima_r)
    w_arima_r = 1 - w_lstm_r
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.lstm_model  import LSTMForecaster, get_lstm
from model.arima_model import ARIMAForecaster, get_arima

MODEL_DIR      = os.path.join(os.path.dirname(__file__), "saved_models")
RESOURCE_NAMES = ["cpu", "memory", "network"]


class CombinedForecaster:
    """
    Per-resource weighted ensemble combining LSTM and ARIMA predictions.

    Each resource dimension receives independent weights based on
    the inverse-RMSE of each model on that specific resource signal.
    """

    def __init__(self):
        self.lstm   = LSTMForecaster()
        self.arima  = ARIMAForecaster()
        # Per-resource weights: {resource_name: weight}
        self.w_lstm  = {"cpu": 0.5, "memory": 0.5, "network": 0.5}
        self.w_arima = {"cpu": 0.5, "memory": 0.5, "network": 0.5}
        self._ready  = False

    def fit(self, data: np.ndarray, verbose: bool = True,
            lstm_metrics: dict | None = None,
            arima_metrics: dict | None = None) -> dict:
        """
        Compute per-resource ensemble weights.

        If metrics from prior training are provided, loads pre-trained
        models and only computes weights. Otherwise trains from scratch.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_steps, 3) multivariate workload data.
        lstm_metrics : dict, optional
            Must contain cpu_rmse, memory_rmse, network_rmse.
        arima_metrics : dict, optional
            Must contain cpu_rmse, memory_rmse, network_rmse.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Handle backward-compatible 1-D data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            data = np.column_stack([data, data * 0.7, data * 0.5])

        # Try to load pre-trained models
        if lstm_metrics is not None and arima_metrics is not None:
            if verbose:
                print("\n  [Combined] Loading pre-trained LSTM and ARIMA...")
            lstm_loaded = self.lstm.load()
            arima_loaded = self.arima.load()
            if not (lstm_loaded and arima_loaded):
                if verbose:
                    print("  [Combined] Pre-trained models not found, training from scratch...")
                lstm_metrics = None
                arima_metrics = None

        if lstm_metrics is None or arima_metrics is None:
            if verbose:
                print("\n  [Combined] Training LSTM component...")
            lstm_metrics = self.lstm.fit(data, verbose=verbose)

            if verbose:
                print("\n  [Combined] Training ARIMA component...")
            arima_metrics = self.arima.fit(data, verbose=verbose)

        # Compute per-resource weights using inverse-RMSE
        for name in RESOURCE_NAMES:
            lstm_rmse  = lstm_metrics.get(f"{name}_rmse", lstm_metrics.get("rmse", 1.0)) + 1e-6
            arima_rmse = arima_metrics.get(f"{name}_rmse", arima_metrics.get("rmse", 1.0)) + 1e-6
            inv_l = 1.0 / lstm_rmse
            inv_a = 1.0 / arima_rmse
            total = inv_l + inv_a
            self.w_lstm[name]  = inv_l / total
            self.w_arima[name] = inv_a / total

        if verbose:
            print(f"\n  [Combined] Per-resource weights:")
            for name in RESOURCE_NAMES:
                print(f"    {name.upper()}: w_lstm={self.w_lstm[name]:.3f}  w_arima={self.w_arima[name]:.3f}")

        # Evaluate combined forecaster on a test segment
        WINDOW = 20
        HORIZON = 5
        test_size = min(200, max(50, int(len(data) * 0.15)))
        eval_start = len(data) - test_size
        n_features = data.shape[1]

        all_preds = []
        all_actuals = []

        for t in range(eval_start, len(data) - HORIZON):
            window = data[max(0, t - WINDOW + 1): t + 1]
            actual = data[t + 1: t + 1 + HORIZON]

            if len(window) < WINDOW or len(actual) < HORIZON:
                continue

            try:
                lstm_pred = self.lstm.predict(window)
            except Exception:
                lstm_pred = np.tile(window[-1], (HORIZON, 1))

            try:
                arima_pred = self.arima.predict(window, steps=HORIZON)
            except Exception:
                arima_pred = np.tile(window[-1], (HORIZON, 1))

            # Ensure shapes match
            if lstm_pred.ndim == 1:
                lstm_pred = lstm_pred.reshape(HORIZON, n_features)
            if arima_pred.ndim == 1:
                arima_pred = arima_pred.reshape(HORIZON, n_features)

            combined_pred = np.zeros_like(lstm_pred)
            for f_idx, name in enumerate(RESOURCE_NAMES[:n_features]):
                combined_pred[:, f_idx] = (
                    self.w_lstm[name]  * lstm_pred[:, f_idx] +
                    self.w_arima[name] * arima_pred[:, f_idx]
                )

            all_preds.append(combined_pred)
            all_actuals.append(actual)

        # Compute metrics from evaluation
        metrics = {}
        if all_preds:
            all_preds_arr = np.array(all_preds)
            all_actuals_arr = np.array(all_actuals)

            all_pred_flat, all_true_flat = [], []
            for f_idx, name in enumerate(RESOURCE_NAMES[:n_features]):
                pred_col = all_preds_arr[:, :, f_idx].flatten()
                true_col = all_actuals_arr[:, :, f_idx].flatten()

                rmse = float(np.sqrt(mean_squared_error(true_col, pred_col)))
                mae  = float(mean_absolute_error(true_col, pred_col))
                r2   = float(r2_score(true_col, pred_col))

                metrics[f"{name}_rmse"] = rmse
                metrics[f"{name}_mae"]  = mae
                metrics[f"{name}_r2"]   = r2

                all_pred_flat.extend(pred_col.tolist())
                all_true_flat.extend(true_col.tolist())

            metrics["rmse"] = float(np.sqrt(mean_squared_error(all_true_flat, all_pred_flat)))
            metrics["mae"]  = float(mean_absolute_error(all_true_flat, all_pred_flat))
            metrics["r2"]   = float(r2_score(all_true_flat, all_pred_flat))
        else:
            # Fallback: use average of component metrics
            metrics["rmse"] = (lstm_metrics.get("rmse", 0) + arima_metrics.get("rmse", 0)) / 2
            metrics["mae"]  = (lstm_metrics.get("mae", 0) + arima_metrics.get("mae", 0)) / 2
            metrics["r2"]   = (lstm_metrics.get("r2", 0) + arima_metrics.get("r2", 0)) / 2

        # Add weight info to metrics
        metrics["w_lstm"]  = self.w_lstm.copy()
        metrics["w_arima"] = self.w_arima.copy()

        if verbose:
            print(f"\n  [Combined] Overall R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}")

        self._ready = True
        return metrics

    def predict(self, window: np.ndarray) -> np.ndarray:
        """
        Produce a combined forecast using per-resource weighted ensemble.

        Parameters
        ----------
        window : np.ndarray
            Shape (window_size, n_features) or 1-D (backward compat).

        Returns
        -------
        np.ndarray
            Shape (forecast_horizon, n_features).
        """
        if not self._ready:
            raise RuntimeError("Combined model not trained.")

        # Backward compatibility: handle 1-D list input
        if isinstance(window, list):
            window = np.array(window, dtype=np.float32)
        if window.ndim == 1:
            w = window[-20:]
            window = np.column_stack([w, w * 0.7, w * 0.5])

        n_features = window.shape[1] if window.ndim > 1 else 3
        HORIZON = 5

        try:
            lstm_pred = self.lstm.predict(window)
        except Exception:
            lstm_pred = np.tile(window[-1], (HORIZON, 1))

        try:
            arima_pred = self.arima.predict(window, steps=HORIZON)
        except Exception:
            arima_pred = np.tile(window[-1], (HORIZON, 1))

        if lstm_pred.ndim == 1:
            lstm_pred = lstm_pred.reshape(HORIZON, n_features)
        if arima_pred.ndim == 1:
            arima_pred = arima_pred.reshape(HORIZON, n_features)

        combined = np.zeros_like(lstm_pred)
        for f_idx, name in enumerate(RESOURCE_NAMES[:n_features]):
            combined[:, f_idx] = (
                self.w_lstm[name]  * lstm_pred[:, f_idx] +
                self.w_arima[name] * arima_pred[:, f_idx]
            )

        return combined

    def save(self, directory: str | None = None):
        d = directory or MODEL_DIR
        os.makedirs(d, exist_ok=True)
        meta = {
            "weights": {
                "lstm":  {k: round(v, 4) for k, v in self.w_lstm.items()},
                "arima": {k: round(v, 4) for k, v in self.w_arima.items()},
            }
        }
        with open(os.path.join(d, "combined_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, directory: str | None = None) -> bool:
        d = directory or MODEL_DIR
        path = os.path.join(d, "combined_meta.json")
        lstm_ok  = self.lstm.load(d)
        arima_ok = self.arima.load(d)
        if os.path.exists(path):
            with open(path) as f:
                meta = json.load(f)
            weights = meta.get("weights", {})
            if "lstm" in weights and isinstance(weights["lstm"], dict):
                self.w_lstm  = weights["lstm"]
                self.w_arima = weights.get("arima", weights.get("var", {}))
            else:
                # Backward compatibility with old scalar format
                wl = meta.get("w_lstm", 0.5)
                wa = meta.get("w_arima", 0.5)
                self.w_lstm  = {"cpu": wl, "memory": wl, "network": wl}
                self.w_arima = {"cpu": wa, "memory": wa, "network": wa}
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
