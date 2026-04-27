# ML Model Documentation

Detailed documentation of all machine learning models used in the predictive scheduling pipeline.

---

## Overview

The project evaluates three proposed forecasting models against a reactive baseline:

| Model | Type | Library | Training Time | Key Files |
|---|---|---|---|---|
| **LSTM** | Deep Learning | PyTorch | ~30s | `model/lstm_model.py`, `model/train_lstm.py` |
| **ARIMA** | Statistical | statsmodels | ~20s | `model/arima_model.py`, `model/train_arima.py` |
| **Combined** | Ensemble | Custom | ~5s | `model/combined_model.py` |
| **GBR** | Tree Ensemble | scikit-learn | ~2s | `model/train_gbr.py` |

---

## 1. LSTM (Long Short-Term Memory)

### Architecture

```
Input: 20 time steps (normalised via MinMaxScaler)
  │
  ▼
LSTM Layer 1 (input_size=1, hidden_size=128, batch_first=True)
  │
  ▼
LSTM Layer 2 (input_size=128, hidden_size=128)
  │
  ▼
Take last hidden state → shape: (batch, 128)
  │
  ▼
BatchNorm1d(128)
  │
  ▼
Linear(128, 64) → ReLU → Dropout(0.15)
  │
  ▼
Linear(64, 32) → ReLU
  │
  ▼
Linear(32, 1) → single output (predicted workload)
```

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Hidden size | 128 | Sufficient for workload pattern complexity |
| Num layers | 2 | Captures multi-level temporal dependencies |
| Window size | 20 | Looks 20 steps back for context |
| Forecast horizon | 5 | Predicts 5 steps into the future |
| Epochs | 150 | Convergence observed around epoch 120 |
| Learning rate | 0.001 | Adam default, stable convergence |
| Batch size | 32 | Fits in CPU memory |
| Dropout | 0.15 | Mild regularisation |
| Loss function | MSE | Standard for regression |

### Training Data Preparation

```python
# In train_all.py → build_full_series()
# Generates ~4500 data points from multiple patterns:
# - gradual(600 pts, seed=42) + gradual(500 pts, seed=17)
# - spike(500 pts, seed=7)    + spike(400 pts, seed=31)
# - periodic(600 pts, seed=3) + periodic(500 pts, seed=55)
# - combined(500 pts, seed=99) + combined(400 pts, seed=11)
```

### Saved Artifacts

| File | Path | Content |
|---|---|---|
| Model weights | `model/saved_models/lstm_model.pt` | PyTorch `state_dict` (~827 KB) |
| Scaler | `model/saved_models/lstm_scaler.pkl` | MinMaxScaler fit on training data |
| Metadata | `model/saved_models/lstm_meta.json` | `{ "hidden_size": 128, "window_size": 20 }` |

### Performance

| Metric | Value |
|---|---|
| **R²** | **0.9696** |
| RMSE | 5.1057 |
| MAE | 4.0363 |

---

## 2. ARIMA (Auto-Regressive Integrated Moving Average)

### Architecture

ARIMA(p, d, q) where:
- **p** = auto-regressive order (number of lagged values)
- **d** = differencing order (to make series stationary)
- **q** = moving average order

### Order Selection

```python
# Grid search: p ∈ [0,5], d ∈ [0,2], q ∈ [0,5]
# Criterion: minimise AIC (Akaike Information Criterion)
# Implementation: statsmodels.tsa.statespace.SARIMAX
# Selected order: typically (2, 1, 2)
```

### Walk-Forward Validation

```
Training segment: 300 steps of combined workload
Split: 80% train, 20% test
For each test step:
  1. Fit ARIMA on expanding window
  2. Predict 1 step ahead
  3. Compare predicted vs actual
  4. Compute R², RMSE, MAE
```

### Saved Artifacts

| File | Path | Content |
|---|---|---|
| Metadata | `model/saved_models/arima_meta.json` | `{ "order": [2,1,2], "aic": 1234.5 }` |

> Note: ARIMA does not save a full model file. It refits on the provided window during inference. Only the order is cached.

### Performance

| Metric | Value |
|---|---|
| **R²** | **0.6351** |
| RMSE | 1.8506 |
| MAE | 0.9267 |

---

## 3. Combined Ensemble (LSTM + ARIMA)

### Weighting Formula

```
w_lstm  = (1 / RMSE_lstm)  / ((1 / RMSE_lstm) + (1 / RMSE_arima))
w_arima = (1 / RMSE_arima) / ((1 / RMSE_lstm) + (1 / RMSE_arima))

prediction = w_lstm × LSTM_prediction + w_arima × ARIMA_prediction
```

### Current Weights

| Weight | Value | Meaning |
|---|---|---|
| w_lstm | 0.434 | ARIMA has lower RMSE → gets slightly more weight |
| w_arima | 0.566 | ARIMA's RMSE (1.85) < LSTM's RMSE (5.11) |

### Evaluation Method

Walk-forward validation with HORIZON=5:
```
For each test index t:
  1. Use series[0:t] as context
  2. LSTM predicts value at t+5
  3. ARIMA predicts value at t+5
  4. Combined = weighted average
  5. Compare with actual[t+5]
```

### Saved Artifacts

| File | Path | Content |
|---|---|---|
| Metadata | `model/saved_models/combined_meta.json` | `{ "w_lstm": 0.434, "w_arima": 0.566 }` |

### Performance

| Metric | Value |
|---|---|
| **R²** | **0.7952** |
| RMSE | 14.5280 |
| MAE | 6.5618 |

---

## 4. GBR (Gradient Boosting Regressor) — Baseline

### Architecture

```
Feature engineering:
  Input: window of 10 lagged values [x_{t-9}, x_{t-8}, ..., x_t]
  Target: x_{t+1} (next value)
  Scaler: MinMaxScaler on features

Model: sklearn.ensemble.GradientBoostingRegressor
  n_estimators = 200
  max_depth = 5
  learning_rate = 0.1
```

### Role in the System

GBR is **not** a proposed research model. It serves as:
1. **Internal inference engine** for the predictive scheduler (fast CPU prediction)
2. **Accuracy baseline** to compare against LSTM/ARIMA/Combined

### Saved Artifacts

| File | Path | Content |
|---|---|---|
| Model | `model/saved_models/gbr_model.pkl` | Serialised GBR model (joblib, ~478 KB) |
| Scaler | `model/saved_models/scaler.pkl` | MinMaxScaler for feature normalisation |

### Performance

| Metric | Value |
|---|---|
| **R²** | **0.9709** |
| RMSE | 4.8521 |
| MAE | 3.3398 |

---

## Training Pipeline

### Master Training Script: `model/train_all.py`

This is the recommended way to train all models:

```bash
python model/train_all.py
```

Execution order:
1. **Build training data** — `build_full_series()` generates ~4500 synthetic points
2. **Train GBR** — scikit-learn, saves to `saved_models/gbr_model.pkl`
3. **Train LSTM** — PyTorch, 150 epochs, saves to `saved_models/lstm_model.pt`
4. **Train ARIMA** — statsmodels, saves to `saved_models/arima_meta.json`
5. **Train Combined** — uses LSTM+ARIMA, saves to `saved_models/combined_meta.json`

### Training via API

Models can also be trained via the Django API:

```bash
# Train a specific model
curl -X POST http://localhost:8000/api/ml/train/ \
  -H "Content-Type: application/json" \
  -d '{"model_type": "lstm"}'

# Train all models
curl -X POST http://localhost:8000/api/ml/train/ \
  -H "Content-Type: application/json" \
  -d '{"model_type": "all"}'
```

Training records are stored in the `ModelTrainingRun` table and visible on the Training page.

---

## Inference Pipeline

### File: `model/inference.py`

Unified prediction interface:

```python
from model.inference import predict

# Single model prediction
result = predict(history=[30.5, 32.1, ...], model_type="lstm")
# Returns: float (predicted workload value)

# Model types: "gbr", "lstm", "arima", "combined"
```

### How the Predictive Scheduler Uses Inference

```python
# In model/predictive_scheduler.py
for each time step:
    history = workload[max(0, t-window):t]
    predicted_cpu = predict(history, model_type="gbr")  # fast inference

    if predicted_cpu > 55:   # scale_up_threshold
        action = "scale_up"
    elif predicted_cpu < 25: # scale_down_threshold
        action = "scale_down"
    else:
        action = "hold"
```

---

## Model Comparison Summary

| Metric | LSTM | ARIMA | Combined | GBR |
|---|---|---|---|---|
| R² | **0.9696** | 0.6351 | 0.7952 | 0.9709 |
| RMSE | 5.11 | **1.85** | 14.53 | 4.85 |
| MAE | 4.04 | **0.93** | 6.56 | 3.34 |
| Training Time | ~30s | ~20s | ~5s | **~2s** |
| Parameters | ~100K | 3 (p,d,q) | 2 (weights) | ~10K |
| GPU Required | No (CPU) | No | No | No |
| Proposed Model | ✓ | ✓ | ✓ | ✗ (baseline) |

### Key Findings
1. **LSTM** achieves the highest R² among proposed models (0.97), demonstrating deep learning's ability to capture non-linear workload patterns
2. **ARIMA** has the lowest RMSE (1.85) despite lower R², indicating good local prediction accuracy
3. **Combined** ensemble (R²=0.80) does not outperform individual models, suggesting the weighting scheme may need refinement
4. **GBR** remains the fastest and most reliable for real-time inference, justifying its use as the scheduler's prediction engine
