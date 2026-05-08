"""
model/lstm_model.py
--------------------
Multi-Input Multi-Output LSTM for workload forecasting across three
cloud resource signals: CPU utilisation, memory consumption, and network I/O.

Architecture:
  Input(window=20, features=3) → LSTM(2-layer, hidden=128)
  → BatchNorm1d(128) → FC(128→64) → FC(64→15)
  → Reshape(5, 3)

The model produces a 5-step-ahead forecast for all three resource
dimensions simultaneously, enabling proactive multi-resource scheduling.

Usage (standalone):
    python model/lstm_model.py

Outputs:
    model/saved_models/lstm_model.pt      (state_dict)
    model/saved_models/lstm_scaler.pkl    (MinMaxScaler fitted on 3 features)
    model/saved_models/lstm_meta.json     (architecture metadata)
"""

import os, sys, json
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WINDOW_SIZE     = 20
FORECAST_HORIZON = 5
N_FEATURES      = 3
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
EPOCHS          = 150
LR              = 0.001
BATCH_SIZE      = 32
MODEL_DIR       = os.path.join(os.path.dirname(__file__), "saved_models")


# ── PyTorch model definition ─────────────────────────────────────────────────
def _get_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class LSTMForecaster:
    """
    Multi-resource LSTM forecaster with batch normalisation.

    Accepts a rolling window of shape (window_size, n_features) and produces
    forecasts of shape (forecast_horizon, n_features).
    """

    def __init__(self, window: int = WINDOW_SIZE, hidden: int = HIDDEN_SIZE,
                 layers: int = NUM_LAYERS, n_features: int = N_FEATURES,
                 forecast_horizon: int = FORECAST_HORIZON):
        self.window           = window
        self.hidden           = hidden
        self.layers           = layers
        self.n_features       = n_features
        self.forecast_horizon = forecast_horizon
        self._model           = None
        self._scaler          = None
        self._ready           = False

    # ── internal helpers ──────────────────────────────────────────────────────
    def _build_model(self):
        torch, nn = _get_torch()

        class _Net(nn.Module):
            """
            Multi-resource LSTM forecasting network.

            Architecture employs a 2-layer LSTM encoder followed by
            batch normalisation and a two-stage fully-connected decoder
            that maps the final hidden state to a flattened
            (forecast_horizon × n_features) output vector.
            """
            def __init__(self, hidden, layers, n_features, forecast_horizon):
                super().__init__()
                self.n_features = n_features
                self.forecast_horizon = forecast_horizon
                self.lstm = nn.LSTM(
                    input_size=n_features, hidden_size=hidden,
                    num_layers=layers, batch_first=True,
                    dropout=0.2 if layers > 1 else 0.0,
                )
                self.bn   = nn.BatchNorm1d(hidden)
                self.fc1  = nn.Linear(hidden, 64)
                self.relu = nn.ReLU()
                self.drop = nn.Dropout(0.15)
                self.fc2  = nn.Linear(64, forecast_horizon * n_features)

            def forward(self, x):
                # x: (B, T, n_features)
                out, _ = self.lstm(x)
                last = out[:, -1, :]       # (B, hidden)
                out = self.bn(last)
                out = self.drop(self.relu(self.fc1(out)))
                out = self.fc2(out)         # (B, forecast_horizon * n_features)
                return out

        return _Net(self.hidden, self.layers, self.n_features, self.forecast_horizon)

    def _build_dataset(self, data: np.ndarray):
        """
        Build sliding-window dataset from multivariate time series.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_timesteps, n_features), already scaled.

        Returns
        -------
        X : np.ndarray, shape (n_samples, window, n_features)
        y : np.ndarray, shape (n_samples, forecast_horizon * n_features)
        """
        X, y = [], []
        for i in range(len(data) - self.window - self.forecast_horizon + 1):
            X.append(data[i: i + self.window])
            target = data[i + self.window: i + self.window + self.forecast_horizon]
            y.append(target.flatten())
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ── training ──────────────────────────────────────────────────────────────
    def fit(self, data: np.ndarray, verbose: bool = True) -> dict:
        """
        Train the LSTM on multivariate workload data.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_timesteps, n_features) or (n_timesteps,) for backward
            compatibility with single-signal data.

        Returns
        -------
        dict
            Per-resource and overall metrics: cpu_r2, memory_r2, network_r2, etc.
        """
        torch, nn = _get_torch()
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Handle backward compatibility: single-signal input
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            data = np.column_stack([data, data * 0.7, data * 0.5])
            self.n_features = 3

        self._scaler = MinMaxScaler()
        data_scaled = self._scaler.fit_transform(data)

        X, y = self._build_dataset(data_scaled)
        split = int(0.85 * len(X))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        # PyTorch tensors
        X_tr_t = torch.tensor(X_tr)
        y_tr_t = torch.tensor(y_tr)
        X_te_t = torch.tensor(X_te)

        model     = self._build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2
        )
        criterion = nn.MSELoss()

        best_loss  = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(X_tr_t))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(X_tr_t), BATCH_SIZE):
                idx = perm[start: start + BATCH_SIZE]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            epoch_loss /= max(1, n_batches)
            scheduler.step(epoch)

            if epoch_loss < best_loss:
                best_loss  = epoch_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping with patience
            if patience_counter > 30:
                if verbose:
                    print(f"  LSTM Early stop at epoch {epoch+1}")
                break

            if verbose and (epoch + 1) % 25 == 0:
                print(f"  LSTM Epoch {epoch+1}/{EPOCHS}  loss={epoch_loss:.6f}  best={best_loss:.6f}")

        model.load_state_dict(best_state)
        self._model = model
        self._ready = True

        # Evaluate on test set: compute per-resource metrics
        model.eval()
        with torch.no_grad():
            y_pred_s = model(X_te_t).numpy()

        # Reshape to (n_samples, forecast_horizon, n_features)
        y_pred_s = y_pred_s.reshape(-1, self.forecast_horizon, self.n_features)
        y_te_r   = y_te.reshape(-1, self.forecast_horizon, self.n_features)

        # Inverse-scale: need to apply inverse per-feature
        resource_names = ["cpu", "memory", "network"]
        metrics = {}
        all_pred, all_true = [], []

        for f_idx in range(self.n_features):
            # Construct dummy arrays for inverse_transform (all other cols zero)
            pred_col = y_pred_s[:, :, f_idx].flatten()
            true_col = y_te_r[:, :, f_idx].flatten()

            # Scale inverse: create n_features-wide arrays
            pred_full = np.zeros((len(pred_col), self.n_features))
            true_full = np.zeros((len(true_col), self.n_features))
            pred_full[:, f_idx] = pred_col
            true_full[:, f_idx] = true_col
            pred_inv = self._scaler.inverse_transform(pred_full)[:, f_idx]
            true_inv = self._scaler.inverse_transform(true_full)[:, f_idx]

            rmse = float(np.sqrt(mean_squared_error(true_inv, pred_inv)))
            mae  = float(mean_absolute_error(true_inv, pred_inv))
            r2   = float(r2_score(true_inv, pred_inv))

            name = resource_names[f_idx] if f_idx < len(resource_names) else f"feat{f_idx}"
            metrics[f"{name}_rmse"] = rmse
            metrics[f"{name}_mae"]  = mae
            metrics[f"{name}_r2"]   = r2

            all_pred.extend(pred_inv.tolist())
            all_true.extend(true_inv.tolist())

        # Overall metrics (average across all resources and forecast steps)
        metrics["rmse"] = float(np.sqrt(mean_squared_error(all_true, all_pred)))
        metrics["mae"]  = float(mean_absolute_error(all_true, all_pred))
        metrics["r2"]   = float(r2_score(all_true, all_pred))

        return metrics

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, window: np.ndarray) -> np.ndarray:
        """
        Forecast the next ``forecast_horizon`` steps for all resources.

        Parameters
        ----------
        window : np.ndarray
            Shape (window_size, n_features) — last ``window_size`` observations.
            Also accepts a 1-D array (single-feature, backward compat).

        Returns
        -------
        np.ndarray
            Shape (forecast_horizon, n_features).
        """
        if not self._ready:
            raise RuntimeError("LSTM not trained. Call fit() or load() first.")
        torch, _ = _get_torch()

        # Backward compatibility: accept 1-D history
        if isinstance(window, list):
            window = np.array(window, dtype=np.float32)
        if window.ndim == 1:
            # Single-signal input: replicate across features for compatibility
            w = window[-self.window:]
            window = np.column_stack([w, w * 0.7, w * 0.5])

        arr = window[-self.window:]
        arr_s = self._scaler.transform(arr)
        x = torch.tensor(arr_s, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
        self._model.eval()
        with torch.no_grad():
            pred_s = self._model(x).numpy().flatten()

        # Reshape and inverse-scale
        pred_s = pred_s.reshape(self.forecast_horizon, self.n_features)
        pred = self._scaler.inverse_transform(pred_s)
        return pred  # (forecast_horizon, n_features)

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, directory: str | None = None):
        torch, _ = _get_torch()
        d = directory or MODEL_DIR
        os.makedirs(d, exist_ok=True)
        torch.save(self._model.state_dict(), os.path.join(d, "lstm_model.pt"))
        joblib.dump(self._scaler, os.path.join(d, "lstm_scaler.pkl"))
        meta = {
            "hidden_size":      self.hidden,
            "window_size":      self.window,
            "n_features":       self.n_features,
            "forecast_horizon": self.forecast_horizon,
            "num_layers":       self.layers,
        }
        with open(os.path.join(d, "lstm_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, directory: str | None = None) -> bool:
        torch, _ = _get_torch()
        d = directory or MODEL_DIR
        meta_path  = os.path.join(d, "lstm_meta.json")
        model_path = os.path.join(d, "lstm_model.pt")
        sc_path    = os.path.join(d, "lstm_scaler.pkl")
        if not all(os.path.exists(p) for p in [model_path, sc_path]):
            return False
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.window           = meta.get("window_size", WINDOW_SIZE)
            self.hidden           = meta.get("hidden_size", HIDDEN_SIZE)
            self.layers           = meta.get("num_layers", NUM_LAYERS)
            self.n_features       = meta.get("n_features", N_FEATURES)
            self.forecast_horizon = meta.get("forecast_horizon", FORECAST_HORIZON)
        self._model = self._build_model()
        self._model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self._model.eval()
        self._scaler = joblib.load(sc_path)
        self._ready  = True
        return True

    def is_ready(self) -> bool:
        return self._ready


# ── Singleton for backend ─────────────────────────────────────────────────────
_lstm_instance: LSTMForecaster | None = None

def get_lstm() -> LSTMForecaster:
    global _lstm_instance
    if _lstm_instance is None:
        _lstm_instance = LSTMForecaster()
        _lstm_instance.load()
    return _lstm_instance
