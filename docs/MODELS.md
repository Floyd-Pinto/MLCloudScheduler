# Model Architecture Documentation

## ML-Based Adaptive Cloud Resource Scheduling — Phase 2

---

## 1. Multi-Resource LSTM

**Class:** `model.lstm_model.LSTMForecaster`

### Architecture
```
Input(20, 3) → LSTM(2-layer, hidden=128, dropout=0.2)
             → BatchNorm1d(128)
             → FC(128 → 64) → ReLU → Dropout(0.15)
             → FC(64 → 15)
             → Reshape(5, 3)
```

### Input / Output
- **Input:** Rolling window of 20 time steps × 3 features (CPU, Memory, Network)
- **Output:** 5-step-ahead forecast for all 3 resources

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Hidden Size | 128 |
| Layers | 2 |
| Batch Size | 32 |
| Epochs | 150 (early stop patience: 30) |
| Learning Rate | 0.001 |
| Scheduler | CosineAnnealingWarmRestarts (T₀=30, T_mult=2) |
| Weight Decay | 1e-5 |
| Gradient Clipping | max_norm=1.0 |

### Scaler
- MinMaxScaler fitted on 3-feature training data
- Saved as `lstm_scaler.pkl`

---

## 2. Per-Channel ARIMA

**Class:** `model.arima_model.ARIMAForecaster`

### Architecture
Independent ARIMA(p, d, q) model per resource channel, each with order
auto-selected via AIC minimisation.

### Order Selection
- Grid search: p ∈ {0,1,2,3}, d ∈ {0,1}, q ∈ {0,1,2,3} → 32 combinations per channel
- Best order chosen by minimum AIC on training split (80/20)
- Walk-forward validation on test split

### Input / Output
- **Input:** History array of shape (n, 3)
- **Output:** (steps, 3) — independent forecast per channel

---

## 3. Multi-Output GBR (Baseline)

**Class:** Trained via `model.train_gbr.train()`

### Architecture
```
MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8
    )
)
```

### Input / Output
- **Input:** Flattened window: 20 × 3 = 60 features
- **Output:** 5 × 3 = 15 target values

### Scaler
- MinMaxScaler fitted on 3-feature training data
- Shared scaler saved as `scaler.pkl`

---

## 4. Combined Ensemble

**Class:** `model.combined_model.CombinedForecaster`

### Weighting Scheme
Per-resource inverse-RMSE weights:

```
For each resource r ∈ {cpu, memory, network}:
    w_lstm_r  = (1/RMSE_lstm_r) / (1/RMSE_lstm_r + 1/RMSE_arima_r)
    w_arima_r = 1 - w_lstm_r

forecast_r = w_lstm_r × LSTM_forecast_r + w_arima_r × ARIMA_forecast_r
```

This enables resource-specific model preference:
- LSTM may dominate CPU prediction (better at capturing non-linear spikes)
- ARIMA may be preferred for memory (smoother, lagged signal)

---

## 5. Anomaly Detector

**Class:** `model.anomaly_detector.AnomalyDetector`

### Dual Strategy
1. **Rolling Z-Score** — sliding window of 30 steps, threshold z > 3.0
2. **Isolation Forest** — 200 trees, contamination 5%, fitted on (n, 3) feature space

An observation is flagged anomalous if **either** detector fires (OR logic),
providing high recall for diverse anomaly types.

### Integration
When anomaly is detected during a predictive scheduler run:
- Scale-up thresholds are temporarily lowered by 10%
- Provides proactive safety margin during unusual events

---

## Resource Signal Correlations

### CPU → Memory
- Temporal lag: ~3 time steps
- Correlation: ρ ≈ 0.75
- Formula: `memory(t) = clip(0.7 × cpu(t-3) + 0.3 × cpu(t) + N(0, 5), 0, 100)`

### CPU → Network I/O
- Proportional with stochastic bursts
- Burst model: Poisson(λ=0.05) amplitude spikes
- Formula: `network(t) = clip(cpu(t) × 0.9 + spike_term + N(0, 8), 0, 100)`

---

## Training Pipeline

```
train_all.py
├── [1/5] GBR            → gbr_model.pkl, scaler.pkl, gbr_meta.json
├── [2/5] LSTM           → lstm_model.pt, lstm_scaler.pkl, lstm_meta.json
├── [3/5] ARIMA          → arima_meta.json, arima_history.pkl
├── [4/5] Combined       → combined_meta.json
├── [5/5] AnomalyDetector → anomaly_iforest.pkl, anomaly_meta.json
└── Summary              → training_summary.json
```
