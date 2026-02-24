"""
schedulers/predictive_scheduler.py
------------------------------------
ML-based predictive scheduler.

Uses a sliding-window feature vector fed into a scikit-learn regressor
(default: GradientBoostingRegressor) to forecast load `horizon` steps
ahead, then scales resources *before* overload occurs.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler


class PredictiveScheduler:
    """
    ML-based predictive scheduler.

    Parameters
    ----------
    window_size          : int   – number of past steps used as features
    horizon              : int   – how many steps ahead to predict
    scale_up_threshold   : float – predicted load above which we scale up
    scale_down_threshold : float – predicted load below which we scale down
    min_capacity         : int   – minimum resource units
    max_capacity         : int   – maximum resource units
    cooldown_steps       : int   – steps between scaling actions
    retrain_every        : int   – retrain model every N steps
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
        self.window_size = window_size
        self.horizon = horizon
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.cooldown_steps = cooldown_steps
        self.retrain_every = retrain_every

        self.capacity = min_capacity
        self._cooldown_counter = 0
        self._history: list[float] = []
        self._step = 0

        self._model = GradientBoostingRegressor(n_estimators=50, max_depth=3,
                                                random_state=42)
        self._scaler = MinMaxScaler()
        self._trained = False

    # ------------------------------------------------------------------
    def _build_dataset(self):
        """Construct (X, y) from the rolling history buffer."""
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

    def _predict(self) -> float | None:
        if not self._trained or len(self._history) < self.window_size:
            return None
        window = np.array(self._history[-self.window_size:]).reshape(1, -1)
        window_scaled = self._scaler.transform(window)
        return float(self._model.predict(window_scaled)[0])

    # ------------------------------------------------------------------
    def observe(self, load: float):
        """Feed the latest load observation into the history buffer."""
        self._history.append(load)
        self._step += 1
        if self._step % self.retrain_every == 0:
            self._train()

    def decide(self) -> int:
        """
        Return the new capacity based on the *predicted* future load.
        Falls back to current capacity if model is not yet trained.
        """
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return self.capacity

        predicted = self._predict()
        if predicted is None:
            return self.capacity

        if predicted > self.scale_up_threshold:
            self.capacity = min(self.capacity + 1, self.max_capacity)
            self._cooldown_counter = self.cooldown_steps
        elif predicted < self.scale_down_threshold:
            self.capacity = max(self.capacity - 1, self.min_capacity)
            self._cooldown_counter = self.cooldown_steps

        return self.capacity

    def reset(self):
        self.capacity = self.min_capacity
        self._cooldown_counter = 0
        self._history.clear()
        self._step = 0
        self._trained = False
