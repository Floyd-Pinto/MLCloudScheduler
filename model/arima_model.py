"""
model/arima_model.py
---------------------
ARIMA forecaster for workload time-series prediction.
Saves trained model params (order + coefficients) for reuse.
"""

import os, sys, json
import numpy as np
import warnings
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
ARIMA_META = os.path.join(MODEL_DIR, "arima_meta.json")

warnings.filterwarnings("ignore")


class ARIMAForecaster:
    """Statsmodels ARIMA wrapper with persistence and multi-step prediction."""

    def __init__(self, order: tuple = (2, 1, 2)):
        self.order  = order
        self._params = None   # fitted params dict
        self._ready  = False
        self._history: list[float] = []

    def fit(self, series: np.ndarray, verbose: bool = True) -> dict:
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Grid-search best ARIMA order on training split
        train_size = int(len(series) * 0.8)
        train = series[:train_size]
        test  = series[train_size:]

        best_aic   = float("inf")
        best_order = self.order

        for p in range(0, 4):
            for d in range(0, 2):
                for q in range(0, 4):
                    try:
                        m = ARIMA(train, order=(p, d, q)).fit()
                        if m.aic < best_aic:
                            best_aic   = m.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue

        if verbose:
            print(f"  ARIMA best order: {best_order}  AIC={best_aic:.2f}")

        self.order = best_order

        # Walk-forward validation on test set
        history    = list(train)
        predictions = []
        for t in range(len(test)):
            try:
                m    = ARIMA(history, order=self.order).fit()
                yhat = m.forecast(steps=1)[0]
            except Exception:
                yhat = history[-1]
            predictions.append(float(yhat))
            history.append(float(test[t]))

        predictions = np.array(predictions)
        rmse = float(np.sqrt(mean_squared_error(test, predictions)))
        mae  = float(mean_absolute_error(test, predictions))
        r2   = float(r2_score(test, predictions))

        # Fit final model on full series
        final_model = ARIMA(series, order=self.order).fit()
        self._params = {
            "order":  list(self.order),
            "params": final_model.params.tolist(),
        }
        self._history = list(series)
        self._ready   = True

        return {"rmse": rmse, "mae": mae, "r2": r2}

    def predict(self, history: list[float], steps: int = 5) -> float:
        """Fit ARIMA on given history and forecast `steps` ahead."""
        if not self._ready:
            raise RuntimeError("ARIMA not trained. Call fit() first.")
        from statsmodels.tsa.arima.model import ARIMA
        try:
            m = ARIMA(history, order=self.order).fit()
            return float(m.forecast(steps=steps).iloc[-1])
        except Exception:
            return float(history[-1])

    def save(self, directory: str | None = None):
        d = directory or MODEL_DIR
        os.makedirs(d, exist_ok=True)
        meta = {"order": list(self.order), "params": self._params}
        with open(os.path.join(d, "arima_meta.json"), "w") as f:
            json.dump(meta, f)

    def load(self, directory: str | None = None) -> bool:
        d = directory or MODEL_DIR
        path = os.path.join(d, "arima_meta.json")
        if not os.path.exists(path):
            return False
        with open(path) as f:
            meta = json.load(f)
        self.order   = tuple(meta["order"])
        self._params = meta.get("params")
        self._ready  = True
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
