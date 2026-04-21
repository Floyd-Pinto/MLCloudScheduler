"""
model/predictive_scheduler.py
------------------------------
ML-based predictive scheduler.
Uses a GradientBoostingRegressor (or a pre-loaded model) to forecast
load `horizon` steps ahead and scales *before* overload occurs.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

from model.metrics_collector import CAPACITY_PER_UNIT

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


class PredictiveScheduler:
    """
    Parameters
    ----------
    window_size          : past steps used as features
    horizon              : steps ahead to predict
    scale_up_threshold   : predicted CPU % above which we scale up
    scale_down_threshold : predicted CPU % below which we scale down
    cooldown_steps       : steps between scaling actions
    retrain_every        : retrain model every N steps
    """

    def __init__(
        self,
        window_size: int = 10,
        horizon: int = 5,
        scale_up_threshold: float = 65.0,
        scale_down_threshold: float = 30.0,
        min_capacity: int = 1,
        max_capacity: int = 20,
        cooldown_steps: int = 5,
        retrain_every: int = 20,
    ):
        self.window_size          = window_size
        self.horizon              = horizon
        self.scale_up_threshold   = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_capacity         = min_capacity
        self.max_capacity         = max_capacity
        self.cooldown_steps       = cooldown_steps
        self.retrain_every        = retrain_every

        self.capacity          = min_capacity
        self._cooldown_counter = 0
        self._history: list[float] = []
        self._step             = 0
        self._model            = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        self._scaler           = MinMaxScaler()
        self._trained          = False

        # Try to load pre-trained model
        self._try_load_pretrained()

    def _try_load_pretrained(self):
        model_path  = os.path.join(MODEL_DIR, "gbr_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self._model   = joblib.load(model_path)
            self._scaler  = joblib.load(scaler_path)
            self._trained = True

    def _build_dataset(self):
        X, y = [], []
        data = np.array(self._history)
        for i in range(len(data) - self.window_size - self.horizon + 1):
            X.append(data[i: i + self.window_size])
            y.append(data[i + self.window_size + self.horizon - 1])
        return np.array(X), np.array(y)

    def _train(self):
        if len(self._history) < self.window_size + self.horizon + 5:
            return
        X, y = self._build_dataset()
        if len(X) < 5:
            return
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._trained = True

    def predict_next(self, history: list[float] | None = None) -> float | None:
        """Public method for API inference."""
        h = history if history is not None else self._history
        if not self._trained or len(h) < self.window_size:
            return None
        window = np.array(h[-self.window_size:]).reshape(1, -1)
        window_scaled = self._scaler.transform(window)
        return float(self._model.predict(window_scaled)[0])

    def observe(self, load: float):
        self._history.append(load)
        self._step += 1
        if self._step % self.retrain_every == 0:
            self._train()

    def decide(self) -> int:
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return self.capacity

        predicted = self.predict_next()

        if predicted is None:
            if not self._history:
                return self.capacity
            current = self._history[-1]
            cpu = (current / max(self.capacity * CAPACITY_PER_UNIT, 1)) * 100.0
        else:
            cpu = (predicted / max(self.capacity * CAPACITY_PER_UNIT, 1)) * 100.0

        if cpu > self.scale_up_threshold:
            self.capacity = min(self.capacity + 1, self.max_capacity)
            self._cooldown_counter = self.cooldown_steps
        elif cpu < self.scale_down_threshold:
            self.capacity = max(self.capacity - 1, self.min_capacity)
            self._cooldown_counter = self.cooldown_steps

        return self.capacity

    def reset(self):
        self.capacity          = self.min_capacity
        self._cooldown_counter = 0
        self._history.clear()
        self._step             = 0
        self._trained          = False
        self._try_load_pretrained()

    def save_model(self, directory: str | None = None):
        if not self._trained:
            return False
        d = directory or MODEL_DIR
        os.makedirs(d, exist_ok=True)
        joblib.dump(self._model,  os.path.join(d, "gbr_model.pkl"))
        joblib.dump(self._scaler, os.path.join(d, "scaler.pkl"))
        return True

    def load_model(self, directory: str | None = None):
        d = directory or MODEL_DIR
        model_path  = os.path.join(d, "gbr_model.pkl")
        scaler_path = os.path.join(d, "scaler.pkl")
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            return False
        self._model   = joblib.load(model_path)
        self._scaler  = joblib.load(scaler_path)
        self._trained = True
        return True
