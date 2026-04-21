"""
model/lstm_model.py
--------------------
Production LSTM for time-series workload forecasting using PyTorch.
Trains a 2-layer LSTM and saves the model checkpoint.

Usage (standalone):
    python model/lstm_model.py

Outputs:
    model/saved_models/lstm_model.pt
    model/saved_models/lstm_scaler.pkl
"""

import os, sys, json
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WINDOW_SIZE = 15
HORIZON     = 5
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
EPOCHS      = 80
LR          = 0.001
BATCH_SIZE  = 64
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "saved_models")


# ── PyTorch model definition ─────────────────────────────────────────────────
def _get_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class LSTMForecaster:
    """Thin wrapper around PyTorch LSTM for use from Django/inference."""

    def __init__(self, window=WINDOW_SIZE, hidden=HIDDEN_SIZE, layers=NUM_LAYERS):
        self.window    = window
        self.hidden    = hidden
        self.layers    = layers
        self._model    = None
        self._scaler   = None
        self._ready    = False

    # ── internal helpers ──────────────────────────────────────────────────────
    def _build_model(self):
        torch, nn = _get_torch()

        class _Net(nn.Module):
            def __init__(self, hidden, layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1, hidden_size=hidden,
                    num_layers=layers, batch_first=True,
                    dropout=0.2 if layers > 1 else 0.0,
                )
                self.fc1  = nn.Linear(hidden, 32)
                self.relu = nn.ReLU()
                self.fc2  = nn.Linear(32, 1)

            def forward(self, x):          # x: (B, T, 1)
                out, _ = self.lstm(x)
                out = self.fc2(self.relu(self.fc1(out[:, -1, :])))
                return out.squeeze(-1)

        return _Net(self.hidden, self.layers)

    def _build_dataset(self, series: np.ndarray):
        X, y = [], []
        for i in range(len(series) - self.window - HORIZON + 1):
            X.append(series[i: i + self.window])
            y.append(series[i + self.window + HORIZON - 1])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ── training ──────────────────────────────────────────────────────────────
    def fit(self, series: np.ndarray, verbose: bool = True) -> dict:
        torch, nn = _get_torch()
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        self._scaler = MinMaxScaler()
        s_scaled = self._scaler.fit_transform(series.reshape(-1, 1)).flatten()

        X, y = self._build_dataset(s_scaled)
        split = int(0.85 * len(X))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        # PyTorch tensors  (B, T, 1)
        X_tr_t = torch.tensor(X_tr).unsqueeze(-1)
        y_tr_t = torch.tensor(y_tr)
        X_te_t = torch.tensor(X_te).unsqueeze(-1)

        model     = self._build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
        criterion = nn.MSELoss()

        best_loss  = float("inf")
        best_state = None

        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(X_tr_t))
            epoch_loss = 0.0
            for start in range(0, len(X_tr_t), BATCH_SIZE):
                idx = perm[start: start + BATCH_SIZE]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= max(1, len(X_tr_t) // BATCH_SIZE)
            scheduler.step(epoch_loss)

            if epoch_loss < best_loss:
                best_loss  = epoch_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  LSTM Epoch {epoch+1}/{EPOCHS}  loss={epoch_loss:.5f}")

        model.load_state_dict(best_state)
        self._model = model
        self._ready = True

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            y_pred_s = model(X_te_t).numpy()

        # Inverse-scale predictions + labels
        y_pred = self._scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        y_true = self._scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))

        return {"rmse": rmse, "mae": mae, "r2": r2}

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, history: list[float]) -> float:
        if not self._ready:
            raise RuntimeError("LSTM not trained. Call fit() or load() first.")
        torch, _ = _get_torch()
        arr = np.array(history[-self.window:], dtype=np.float32)
        arr_s = self._scaler.transform(arr.reshape(-1, 1)).flatten()
        x = torch.tensor(arr_s).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        self._model.eval()
        with torch.no_grad():
            pred_s = self._model(x).item()
        return float(self._scaler.inverse_transform([[pred_s]])[0][0])

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, directory: str | None = None):
        torch, _ = _get_torch()
        d = directory or MODEL_DIR
        os.makedirs(d, exist_ok=True)
        torch.save(self._model.state_dict(), os.path.join(d, "lstm_model.pt"))
        joblib.dump(self._scaler, os.path.join(d, "lstm_scaler.pkl"))
        meta = {"window": self.window, "hidden": self.hidden, "layers": self.layers}
        with open(os.path.join(d, "lstm_meta.json"), "w") as f:
            json.dump(meta, f)

    def load(self, directory: str | None = None):
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
            self.window = meta.get("window", WINDOW_SIZE)
            self.hidden = meta.get("hidden", HIDDEN_SIZE)
            self.layers = meta.get("layers", NUM_LAYERS)
        self._model = self._build_model()
        self._model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
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
