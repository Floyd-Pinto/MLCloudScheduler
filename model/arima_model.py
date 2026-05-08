"""
model/arima_model.py
---------------------
Multi-resource ARIMA forecaster for workload time-series prediction.

Fits independent ARIMA models per resource dimension (CPU, memory,
network I/O), each with automatic order selection via AIC grid search.
This preserves the proven univariate ARIMA methodology while extending
it to the three-resource setting required by Phase 2.

Each channel's (p, d, q) order is selected independently, reflecting
the distinct temporal dynamics of each resource signal.
"""

import os, sys, json
import numpy as np
import warnings
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "saved_models")
ARIMA_META = os.path.join(MODEL_DIR, "arima_meta.json")

# Suppress statsmodels convergence warnings during ARIMA grid search
# These are expected: many (p,d,q) combinations intentionally fail
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood.*")

RESOURCE_NAMES = ["cpu", "memory", "network"]


class ARIMAForecaster:
    """
    Multi-resource ARIMA wrapper with per-channel order selection,
    persistence, and multi-step prediction.

    Fits one statsmodels ARIMA model per resource dimension. During
    inference, each channel is forecast independently and the results
    are stacked into a (steps, n_features) array.
    """

    def __init__(self, order: tuple = (2, 1, 2), n_features: int = 3):
        self.default_order = order
        self.n_features    = n_features
        # Per-channel fitted orders
        self._orders: list[tuple] = [order] * n_features
        self._ready  = False
        self._history: np.ndarray | None = None   # (n_steps, n_features)

    def fit(self, data: np.ndarray, verbose: bool = True) -> dict:
        """
        Fit independent ARIMA models per resource channel.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_timesteps, n_features) or (n_timesteps,) for a
            single signal (backward compatible).

        Returns
        -------
        dict
            Per-resource and overall R², RMSE, MAE.
        """
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Backward compatibility: promote 1-D to 2-D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.n_features = data.shape[1]
        self._orders = [self.default_order] * self.n_features

        train_size = int(len(data) * 0.8)
        train = data[:train_size]
        test  = data[train_size:]

        metrics = {}
        all_pred, all_true = [], []

        for f_idx in range(self.n_features):
            series_train = train[:, f_idx]
            series_test  = test[:, f_idx]

            # Grid-search best ARIMA order on training split
            best_aic   = float("inf")
            best_order = self.default_order

            for p in range(0, 4):
                for d in range(0, 2):
                    for q in range(0, 4):
                        try:
                            m = ARIMA(series_train, order=(p, d, q)).fit()
                            if m.aic < best_aic:
                                best_aic   = m.aic
                                best_order = (p, d, q)
                        except Exception:
                            continue

            self._orders[f_idx] = best_order
            name = RESOURCE_NAMES[f_idx] if f_idx < len(RESOURCE_NAMES) else f"feat{f_idx}"

            if verbose:
                print(f"  ARIMA [{name}] best order: {best_order}  AIC={best_aic:.2f}")

            # Walk-forward validation on test set
            history = list(series_train)
            predictions = []
            for t in range(min(len(series_test), 300)):
                try:
                    m    = ARIMA(history, order=best_order).fit()
                    yhat = m.forecast(steps=1)[0]
                except Exception:
                    yhat = history[-1]
                predictions.append(float(yhat))
                history.append(float(series_test[t]))

            predictions = np.array(predictions)
            actual = series_test[:len(predictions)]

            rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
            mae  = float(mean_absolute_error(actual, predictions))
            r2   = float(r2_score(actual, predictions))

            metrics[f"{name}_rmse"] = rmse
            metrics[f"{name}_mae"]  = mae
            metrics[f"{name}_r2"]   = r2

            all_pred.extend(predictions.tolist())
            all_true.extend(actual.tolist())

        # Overall metrics
        metrics["rmse"] = float(np.sqrt(mean_squared_error(all_true, all_pred)))
        metrics["mae"]  = float(mean_absolute_error(all_true, all_pred))
        metrics["r2"]   = float(r2_score(all_true, all_pred))

        self._history = data.copy()
        self._ready   = True

        return metrics

    def predict(self, history: np.ndarray, steps: int = 5) -> np.ndarray:
        """
        Forecast ``steps`` ahead for all resources using per-channel ARIMA.

        Parameters
        ----------
        history : np.ndarray
            Shape (window, n_features) or 1-D list (backward compat).
        steps : int
            Number of steps to forecast.

        Returns
        -------
        np.ndarray
            Shape (steps, n_features).
        """
        if not self._ready:
            raise RuntimeError("ARIMA not trained. Call fit() first.")
        from statsmodels.tsa.arima.model import ARIMA

        # Handle backward compatibility: 1-D input
        if isinstance(history, list):
            history = np.array(history, dtype=np.float64)
        if history.ndim == 1:
            history = history.reshape(-1, 1)
            if self.n_features > 1:
                # Pad with correlated signals for compatibility
                cols = [history]
                for _ in range(self.n_features - 1):
                    cols.append(history * 0.7)
                history = np.column_stack(cols)

        n_features = min(history.shape[1], self.n_features)
        result = np.zeros((steps, self.n_features))

        for f_idx in range(n_features):
            series = history[:, f_idx].tolist()
            order  = self._orders[f_idx] if f_idx < len(self._orders) else self.default_order
            try:
                m = ARIMA(series, order=order).fit()
                forecast = m.forecast(steps=steps)
                result[:, f_idx] = np.clip(forecast.values if hasattr(forecast, 'values') else forecast, 0, 100)
            except Exception:
                # Fallback: repeat last value
                result[:, f_idx] = series[-1]

        return result

    def save(self, directory: str | None = None):
        d = directory or MODEL_DIR
        os.makedirs(d, exist_ok=True)
        meta = {
            "model_type": "ARIMA",
            "orders":     [list(o) for o in self._orders],
            "n_features": self.n_features,
        }
        with open(os.path.join(d, "arima_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        # Save history for prediction context
        if self._history is not None:
            joblib.dump(self._history, os.path.join(d, "arima_history.pkl"))

    def load(self, directory: str | None = None) -> bool:
        d = directory or MODEL_DIR
        path = os.path.join(d, "arima_meta.json")
        if not os.path.exists(path):
            return False
        with open(path) as f:
            meta = json.load(f)
        self.n_features = meta.get("n_features", 3)
        orders = meta.get("orders")
        if orders:
            self._orders = [tuple(o) for o in orders]
        else:
            # Backward compatibility: single order
            order = meta.get("order", [2, 1, 2])
            self._orders = [tuple(order)] * self.n_features

        # Load history if available
        hist_path = os.path.join(d, "arima_history.pkl")
        if os.path.exists(hist_path):
            self._history = joblib.load(hist_path)

        self._ready = True
        return True

    def is_ready(self) -> bool:
        return self._ready


# ── Singleton ─────────────────────────────────────────────────────────────────
_arima_instance: ARIMAForecaster | None = None

def get_arima() -> ARIMAForecaster:
    global _arima_instance
    if _arima_instance is None:
        _arima_instance = ARIMAForecaster()
        _arima_instance.load()
    return _arima_instance
