"""
model/anomaly_detector.py
--------------------------
Anomaly detection module for multi-resource workload telemetry.

Combines two complementary detection strategies:
  1. Rolling Z-Score — detects sudden deviations within a sliding window
  2. Isolation Forest — detects structural anomalies in the 3-D feature space

An observation is flagged as anomalous if EITHER detector fires, providing
high recall for diverse anomaly types (point anomalies, contextual anomalies).

Integration with PredictiveScheduler:
  When an anomaly is detected, the scheduler temporarily lowers its
  scale-up thresholds by 10% to provide a proactive safety margin.
"""

import os
import numpy as np
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

RESOURCE_NAMES = ["cpu", "memory", "network"]


class AnomalyDetector:
    """
    Multi-resource anomaly detector using Rolling Z-Score + Isolation Forest.

    Parameters
    ----------
    z_threshold : float
        Z-score threshold beyond which a point is flagged (default: 3.0).
    rolling_window : int
        Number of recent steps for computing rolling statistics.
    contamination : float
        Expected proportion of anomalies for Isolation Forest (default: 0.05).
    """

    def __init__(self, z_threshold: float = 3.0, rolling_window: int = 30,
                 contamination: float = 0.05):
        self.z_threshold    = z_threshold
        self.rolling_window = rolling_window
        self.contamination  = contamination
        self._iso_forest    = None
        self._history       = []       # list of [cpu, mem, net] for rolling stats
        self._fitted        = False

    def fit(self, data: np.ndarray, verbose: bool = True) -> dict:
        """
        Fit the Isolation Forest on multivariate training data.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_steps, 3).

        Returns
        -------
        dict
            Number of anomalies detected in training set (for diagnostics).
        """
        from sklearn.ensemble import IsolationForest

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._iso_forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        self._iso_forest.fit(data)

        # Diagnostics: count training anomalies
        train_preds = self._iso_forest.predict(data)
        n_anomalies = int((train_preds == -1).sum())
        n_total = len(data)

        self._fitted = True

        result = {
            "n_training_samples": n_total,
            "n_anomalies_detected": n_anomalies,
            "anomaly_rate": round(n_anomalies / max(n_total, 1) * 100, 2),
        }

        if verbose:
            print(f"  [AnomalyDetector] Fitted on {n_total} samples. "
                  f"Training anomalies: {n_anomalies} ({result['anomaly_rate']}%)")

        return result

    def detect(self, observation: np.ndarray) -> bool:
        """
        Check if the given observation is anomalous.

        Parameters
        ----------
        observation : np.ndarray
            Shape (1, 3) or (3,) — single multi-resource observation.

        Returns
        -------
        bool
            True if the observation is anomalous.
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        self._history.append(observation.flatten().tolist())

        # Strategy 1: Rolling Z-Score
        z_flag = False
        if len(self._history) >= self.rolling_window:
            recent = np.array(self._history[-self.rolling_window:])
            means = recent.mean(axis=0)
            stds  = recent.std(axis=0) + 1e-8
            z_scores = np.abs((observation.flatten() - means) / stds)
            z_flag = bool(np.any(z_scores > self.z_threshold))

        # Strategy 2: Isolation Forest
        iso_flag = False
        if self._iso_forest is not None:
            pred = self._iso_forest.predict(observation)
            iso_flag = bool(pred[0] == -1)

        return z_flag or iso_flag

    def detect_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Detect anomalies in a batch of observations.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_steps, 3).

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_steps,) — True where anomalous.
        """
        flags = np.zeros(len(data), dtype=bool)
        for i, obs in enumerate(data):
            flags[i] = self.detect(obs)
        return flags

    def save(self, directory: str | None = None):
        d = directory or MODEL_DIR
        os.makedirs(d, exist_ok=True)
        if self._iso_forest is not None:
            joblib.dump(self._iso_forest, os.path.join(d, "anomaly_iforest.pkl"))
        meta = {
            "z_threshold":    self.z_threshold,
            "rolling_window": self.rolling_window,
            "contamination":  self.contamination,
        }
        import json
        with open(os.path.join(d, "anomaly_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, directory: str | None = None) -> bool:
        d = directory or MODEL_DIR
        iso_path  = os.path.join(d, "anomaly_iforest.pkl")
        meta_path = os.path.join(d, "anomaly_meta.json")
        if not os.path.exists(iso_path):
            return False
        self._iso_forest = joblib.load(iso_path)
        if os.path.exists(meta_path):
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            self.z_threshold    = meta.get("z_threshold", 3.0)
            self.rolling_window = meta.get("rolling_window", 30)
            self.contamination  = meta.get("contamination", 0.05)
        self._fitted = True
        return True

    def is_ready(self) -> bool:
        return self._fitted

    def reset(self):
        """Reset rolling history (for new simulation runs)."""
        self._history = []
