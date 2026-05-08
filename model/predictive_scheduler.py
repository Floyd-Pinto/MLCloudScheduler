"""
model/predictive_scheduler.py
------------------------------
Multi-resource ML-based predictive scheduler with anomaly-aware
threshold adjustment.

Uses the Combined (LSTM+ARIMA) ensemble to forecast 5 steps ahead
for all three resources. Scales up proactively when ANY forecasted
resource exceeds its threshold, and includes a rate-of-change
spike-detection fast-path for sudden workload surges.

Anomaly integration:
  If an AnomalyDetector is available, detected anomalies temporarily
  lower the scale-up thresholds by 10% to provide a safety margin.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.metrics_collector import CAPACITY_PER_UNIT

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


class PredictiveScheduler:
    """
    Multi-resource predictive scheduler with ML-based forecasting.

    Parameters
    ----------
    window_size      : rolling window of observations kept for prediction
    horizon          : forecast horizon (steps ahead)
    cpu_up_thresh    : CPU threshold for proactive scale-up
    mem_up_thresh    : Memory threshold for proactive scale-up
    net_up_thresh    : Network I/O threshold for proactive scale-up
    scale_down_threshold : All resources below this → scale-in candidate
    cooldown_steps   : minimum steps between scaling actions
    min_capacity     : minimum resource units
    max_capacity     : maximum resource units
    """

    def __init__(
        self,
        window_size: int = 20,
        horizon: int = 5,
        cpu_up_thresh: float = 65.0,
        mem_up_thresh: float = 65.0,
        net_up_thresh: float = 70.0,
        scale_down_threshold: float = 30.0,
        cooldown_steps: int = 3,
        min_capacity: int = 1,
        max_capacity: int = 20,
        retrain_every: int = 10,
        # Legacy parameter for backward compat
        scale_up_threshold: float | None = None,
    ):
        self.window_size   = window_size
        self.horizon       = horizon
        self.cpu_up_thresh = cpu_up_thresh
        self.mem_up_thresh = mem_up_thresh
        self.net_up_thresh = net_up_thresh
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_steps = cooldown_steps
        self.min_capacity  = min_capacity
        self.max_capacity  = max_capacity
        self.retrain_every = retrain_every

        self.capacity      = min_capacity
        self._history      = []       # list of [cpu, mem, net] observations
        self._cooldown     = 0
        self._model        = None
        self._anomaly_det  = None
        self._last_trigger = "none"

    def _load_model(self):
        """Lazy-load the combined forecaster."""
        if self._model is not None:
            return
        try:
            from model.combined_model import CombinedForecaster
            m = CombinedForecaster()
            if m.load():
                self._model = m
        except Exception:
            pass

    def _load_anomaly_detector(self):
        """Lazy-load anomaly detector if available."""
        if self._anomaly_det is not None:
            return
        try:
            from model.anomaly_detector import AnomalyDetector
            det = AnomalyDetector()
            if det.load():
                self._anomaly_det = det
        except Exception:
            pass

    def observe(self, load: float, cpu_pct: float | None = None,
                mem_pct: float | None = None, net_pct: float | None = None):
        """
        Record a single step of multi-resource observations.

        Parameters
        ----------
        load : float
            Raw workload value.
        cpu_pct, mem_pct, net_pct : float, optional
            Explicit utilisation percentages. If not provided,
            derived from load and capacity.
        """
        if cpu_pct is None:
            cpu_pct = min((load / max(self.capacity * CAPACITY_PER_UNIT, 1)) * 100.0, 100.0)
        if mem_pct is None:
            mem_pct = cpu_pct * 0.7
        if net_pct is None:
            net_pct = cpu_pct * 0.5

        self._history.append([cpu_pct, mem_pct, net_pct])

    def decide(self) -> int:
        """
        Make a proactive scaling decision using ML forecasts.

        Returns
        -------
        int
            Updated capacity.
        """
        self._load_model()
        self._load_anomaly_detector()

        if self._cooldown > 0:
            self._cooldown -= 1

        # Need enough history for the forecaster window
        if len(self._history) < self.window_size:
            return self.capacity

        window = np.array(self._history[-self.window_size:], dtype=np.float32)

        # Rate-of-change fast-path: detect spike onset via acceleration
        if len(self._history) >= 4:
            recent = [h[0] for h in self._history[-4:]]  # CPU signal
            grad = np.diff(recent)
            accel = np.diff(grad)
            cap_fraction = self.capacity * CAPACITY_PER_UNIT
            if any(a > 0.15 * cap_fraction for a in accel):
                if self._cooldown == 0:
                    self.capacity = min(self.capacity + 1, self.max_capacity)
                    self._cooldown = self.cooldown_steps
                    self._last_trigger = "spike_fast_path"
                    return self.capacity

        # Anomaly-aware threshold adjustment
        cpu_thresh = self.cpu_up_thresh
        mem_thresh = self.mem_up_thresh
        net_thresh = self.net_up_thresh

        if self._anomaly_det is not None:
            try:
                latest = window[-1:]  # (1, 3)
                is_anomaly = self._anomaly_det.detect(latest)
                if is_anomaly:
                    # Lower thresholds by 10% during anomalies
                    cpu_thresh *= 0.9
                    mem_thresh *= 0.9
                    net_thresh *= 0.9
            except Exception:
                pass

        # ML-based forecasting
        should_scale_up  = False
        should_scale_down = False
        trigger = "none"

        if self._model is not None and self._model.is_ready():
            try:
                forecast = self._model.predict(window)  # (horizon, 3)
                # Check if any forecasted resource exceeds threshold
                cpu_forecast = forecast[:, 0]
                mem_forecast = forecast[:, 1]
                net_forecast = forecast[:, 2]

                if np.any(cpu_forecast > cpu_thresh):
                    should_scale_up = True
                    trigger = "cpu_forecast"
                if np.any(mem_forecast > mem_thresh):
                    should_scale_up = True
                    trigger = trigger + ",mem_forecast" if trigger != "none" else "mem_forecast"
                if np.any(net_forecast > net_thresh):
                    should_scale_up = True
                    trigger = trigger + ",net_forecast" if trigger != "none" else "net_forecast"

                # Scale-down: all forecasted values are low
                if (np.all(cpu_forecast < self.scale_down_threshold) and
                    np.all(mem_forecast < self.scale_down_threshold) and
                    np.all(net_forecast < self.scale_down_threshold)):
                    should_scale_down = True
                    trigger = "all_low_forecast"
            except Exception:
                pass
        else:
            # Fallback: use last known values as naive forecast
            last = window[-1]
            if (last[0] > cpu_thresh or last[1] > mem_thresh or last[2] > net_thresh):
                should_scale_up = True
                trigger = "threshold_fallback"
            if all(v < self.scale_down_threshold for v in last):
                should_scale_down = True
                trigger = "all_low_fallback"

        self._last_trigger = trigger

        # Apply decisions with cooldown
        if should_scale_up and self._cooldown == 0:
            self.capacity = min(self.capacity + 1, self.max_capacity)
            self._cooldown = self.cooldown_steps
        elif should_scale_down and not should_scale_up and self._cooldown == 0:
            self.capacity = max(self.capacity - 1, self.min_capacity)
            self._cooldown = self.cooldown_steps

        return self.capacity

    @property
    def last_trigger(self) -> str:
        return self._last_trigger

    def reset(self):
        self.capacity  = self.min_capacity
        self._history  = []
        self._cooldown = 0
        self._last_trigger = "none"
