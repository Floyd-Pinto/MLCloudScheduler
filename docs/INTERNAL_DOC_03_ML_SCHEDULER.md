# MASTER INTERNAL DOCUMENTATION — Part 3
# Sections 5–6: ML System & Scheduler Documentation

---

# SECTION 5 — MACHINE LEARNING SYSTEM DOCUMENTATION

## 5.1 LSTM Architecture

**Class**: `model.lstm_model.LSTMForecaster`

```
Input(batch, 20, 3)
  → LSTM(input_size=3, hidden_size=128, num_layers=2, dropout=0.2, batch_first=True)
  → Take last hidden state: (batch, 128)
  → BatchNorm1d(128)
  → Linear(128 → 64) → ReLU → Dropout(0.15)
  → Linear(64 → 15)
  → Reshape to (batch, 5, 3)
Output: 5-step forecast for [CPU, Memory, Network]
```

### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window size | 20 | Captures medium-range temporal patterns |
| Forecast horizon | 5 | Balances prediction length vs accuracy |
| Hidden size | 128 | Sufficient capacity for 3-feature input |
| LSTM layers | 2 | Captures hierarchical temporal abstractions |
| LSTM dropout | 0.2 | Regularisation between LSTM layers |
| FC dropout | 0.15 | Regularisation after decoder |
| Batch size | 32 | Standard mini-batch size |
| Epochs | 150 | With early stopping (patience=30) |
| Learning rate | 0.001 | Standard Adam default |
| Weight decay | 1e-5 | L2 regularisation |
| LR scheduler | CosineAnnealingWarmRestarts(T₀=30, T_mult=2) | Cyclical LR for escaping local minima |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients |
| Train/test split | 85/15 | — |

### Training Data
- 4,200 steps from 9 diverse workload segments (varied patterns + seeds)
- MinMaxScaler fitted on 3-feature training data → saved as `lstm_scaler.pkl`

### Per-Resource Results (Phase 2)
| Resource | R² | RMSE | MAE |
|----------|-----|------|-----|
| CPU | 0.9632 | 5.777 | 3.605 |
| Memory | 0.9520 | 6.454 | 4.434 |
| Network | 0.8657 | 10.577 | 7.940 |
| **Overall** | **0.9287** | **7.893** | **5.326** |

### Why LSTM?
- Captures non-linear temporal dependencies in workload patterns
- State-of-the-art for sequence forecasting tasks
- Multi-input/output architecture handles 3 resources simultaneously
- BatchNorm stabilises training with varied workload magnitudes

---

## 5.2 ARIMA Architecture

**Class**: `model.arima_model.ARIMAForecaster`

### Design
- **3 independent ARIMA models**, one per resource channel
- Each channel gets its own (p,d,q) order via AIC minimisation
- Grid search: p∈{0,1,2,3}, d∈{0,1}, q∈{0,1,2,3} → 32 combinations/channel
- Walk-forward validation on 300-step test segment

### Per-Channel Orders
Orders auto-selected per channel; stored in `arima_meta.json`.

### Per-Resource Results (Phase 2)
| Resource | R² | RMSE | MAE |
|----------|-----|------|-----|
| CPU | 0.8741 | 3.285 | 2.498 |
| Memory | 0.5911 | 6.482 | 5.028 |
| Network | 0.2185 | 9.980 | 7.907 |
| **Overall** | **0.5624** | **7.127** | **5.144** |

### Why ARIMA?
- Established statistical baseline for time-series forecasting
- Handles linear trends and seasonality explicitly via differencing
- Interpretable model with well-understood theory
- ARIMA excels on CPU (linear trend component) but struggles with bursty Network I/O

---

## 5.3 GBR Architecture

**Training**: `model.train_gbr.train()`

```
MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=200, max_depth=5,
        learning_rate=0.05, subsample=0.8,
        random_state=42
    ),
    n_jobs=-1
)
```

### Input/Output
- **Input**: Flattened window: 20 steps × 3 features = **60 features**
- **Output**: 5 steps × 3 features = **15 target values**
- Internally: 15 independent GBR models (one per output dimension)

### Training Data
- 3,300 steps from 7 workload segments
- MinMaxScaler shared with inference (saved as `scaler.pkl`)
- 80/20 train/test split

### Per-Resource Results (Phase 2)
| Resource | R² | RMSE | MAE |
|----------|-----|------|-----|
| CPU | 0.9638 | 5.303 | 2.842 |
| Memory | 0.9427 | 6.634 | 4.536 |
| Network | 0.8341 | 10.943 | 8.106 |
| **Overall** | **0.9158** | **7.997** | **5.161** |

### Why GBR?
- Fast inference (no GPU needed) — ideal for real-time scheduler
- Competitive accuracy with LSTM on tabular/windowed data
- Robust to noise and non-linear relationships
- **Used as the baseline model**, not the primary research contribution

---

## 5.4 Combined Ensemble

**Class**: `model.combined_model.CombinedForecaster`

### Per-Resource Inverse-RMSE Weighting
```
For each resource r ∈ {cpu, memory, network}:
    w_lstm_r  = (1/RMSE_lstm_r) / (1/RMSE_lstm_r + 1/RMSE_arima_r)
    w_arima_r = 1 - w_lstm_r
    
    forecast_r = w_lstm_r × LSTM_forecast_r + w_arima_r × ARIMA_forecast_r
```

### Computed Weights (from latest training)
| Resource | w_lstm | w_arima |
|----------|--------|---------|
| CPU | 0.363 | 0.637 |
| Memory | 0.501 | 0.499 |
| Network | 0.485 | 0.515 |

**Key insight**: ARIMA gets higher CPU weight because its CPU RMSE (3.29) is lower than LSTM's (5.78), despite LSTM having higher overall R². This is the per-resource advantage.

### Per-Resource Results (Phase 2)
| Resource | R² | RMSE | MAE |
|----------|-----|------|-----|
| CPU | 0.6188 | 3.494 | 2.515 |
| Memory | 0.3134 | 4.528 | 3.638 |
| Network | 0.0907 | 8.236 | 6.648 |
| **Overall** | **0.4698** | **5.789** | **4.267** |

### Why Combined Underperforms
- Evaluated on a **separate test segment** where both sub-models introduce error
- Error compounding: weighting two noisy predictions doesn't reduce variance
- Phase shift between LSTM and ARIMA predictions on test data causes destructive interference
- **This is a valid research finding**: ensemble ≠ always better

---

## 5.5 Anomaly Detection System

**Class**: `model.anomaly_detector.AnomalyDetector`

### Dual Strategy (OR Logic)

**Strategy 1: Rolling Z-Score**
```
For each step t:
    recent = history[t-30 : t]  # sliding window of 30 steps
    means = recent.mean(axis=0)  # per-resource means
    stds  = recent.std(axis=0)   # per-resource stds
    z_scores = |observation - means| / stds
    z_flag = any(z_scores > 3.0)
```

**Strategy 2: Isolation Forest**
```
IsolationForest(
    n_estimators=200,
    contamination=0.05,  # expect 5% anomalies
    random_state=42,
    n_jobs=-1
)
Fitted on (n_steps, 3) multi-resource feature space
iso_flag = (iforest.predict(observation) == -1)
```

**Combined**: `is_anomaly = z_flag OR iso_flag`

- **OR logic provides high recall** for diverse anomaly types
- Z-Score catches **point anomalies** (sudden deviations)
- IsolationForest catches **structural anomalies** (unusual combinations)

### Training Diagnostics
- 600 training samples
- 30 anomalies detected (5.0% rate — matches contamination parameter)

### Integration with Scheduler
When anomaly detected during predictive scheduler step:
- CPU threshold: 65% → 58.5% (×0.9)
- Memory threshold: 65% → 58.5% (×0.9)
- Network threshold: 70% → 63% (×0.9)
- **Provides proactive safety margin during unusual events**

---

## 5.6 Model Comparison Summary

| Model | Overall R² | Overall RMSE | CPU R² | Mem R² | Net R² | Inference Speed | Role |
|-------|-----------|-------------|--------|--------|--------|----------------|------|
| **LSTM** | **0.929** | 7.89 | 0.963 | 0.952 | 0.866 | ~10ms | Research model |
| **GBR** | 0.916 | 8.00 | 0.964 | 0.943 | 0.834 | ~1ms | Scheduler backbone |
| **ARIMA** | 0.562 | 7.13 | 0.874 | 0.591 | 0.219 | ~50ms | Statistical baseline |
| **Combined** | 0.470 | 5.79 | 0.619 | 0.313 | 0.091 | ~60ms | Ensemble experiment |

### Why GBR Became the Scheduler Inference Model
The predictive scheduler actually uses **CombinedForecaster** (LSTM+ARIMA ensemble) for forecasting, not GBR directly. However, GBR serves as the fast **baseline reference model** and is available for fallback. The combined model is used because it tests the research hypothesis about ensemble benefit.

### Model Artifacts Pipeline
```
train_all.py
├── [1/5] GBR            → gbr_model.pkl, scaler.pkl, gbr_meta.json
├── [2/5] LSTM           → lstm_model.pt, lstm_scaler.pkl, lstm_meta.json
├── [3/5] ARIMA          → arima_meta.json, arima_history.pkl
├── [4/5] Combined       → combined_meta.json
├── [5/5] AnomalyDetector → anomaly_iforest.pkl, anomaly_meta.json
└── Summary              → training_summary.json
```

---

# SECTION 6 — SCHEDULER SYSTEM DOCUMENTATION

## 6.1 Reactive Scheduler

**Class**: `model.reactive_scheduler.ReactiveScheduler`

### Decision Logic Pseudocode
```
function decide(load, cpu_pct, mem_pct, net_pct):
    # Derive utilisation if not provided
    if cpu_pct is null:
        cpu_pct = min(load / (capacity × 10) × 100, 100)
        mem_pct = cpu_pct × 0.7
        net_pct = cpu_pct × 0.5
    
    # Check per-resource overloads
    any_overload = (cpu_pct > 80) OR (mem_pct > 80) OR (net_pct > 85)
    
    # Track consecutive overload steps
    if any_overload:
        overload_counter++
        low_counter = 0
    else:
        overload_counter = 0
    
    # Track consecutive low-utilisation steps
    all_low = (cpu_pct < 40) AND (mem_pct < 40) AND (net_pct < 40)
    if all_low: low_counter++
    else: low_counter = 0
    
    # Scale out: 2 consecutive overload steps
    if overload_counter >= 2:
        capacity = min(capacity + 1, 20)
        overload_counter = 0
    
    # Scale in: 5 consecutive low steps
    elif low_counter >= 5:
        capacity = max(capacity - 1, 1)
        low_counter = 0
    
    return capacity
```

### Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| cpu_threshold | 80% | Standard cloud overload threshold |
| mem_threshold | 80% | Matches CPU threshold |
| net_threshold | 85% | Slightly higher (network more tolerant) |
| scale_down_threshold | 40% | Conservative scale-in point |
| consec_up | 2 | Avoids single-step noise triggering scale-out |
| consec_down | 5 | Conservative scale-in (avoid thrashing) |
| min_capacity | 1 | Minimum 1 resource unit |
| max_capacity | 20 | Upper bound |

## 6.2 Predictive Scheduler

**Class**: `model.predictive_scheduler.PredictiveScheduler`

### Decision Logic Pseudocode
```
function decide():
    load_model()  # lazy-load CombinedForecaster
    load_anomaly_detector()  # lazy-load AnomalyDetector
    
    if cooldown > 0: cooldown--
    if len(history) < 20: return capacity  # not enough data
    
    window = history[-20:]  # (20, 3)
    
    # FAST PATH: Rate-of-change spike detection
    if len(history) >= 4:
        recent_cpu = history[-4:][cpu]
        gradient = diff(recent_cpu)
        acceleration = diff(gradient)
        if any(acceleration > 0.15 × capacity × 10):
            if cooldown == 0:
                capacity++; cooldown = 3
                return capacity  # immediate response to spike onset
    
    # ANOMALY-AWARE THRESHOLD ADJUSTMENT
    cpu_thresh, mem_thresh, net_thresh = 65, 65, 70
    if anomaly_detector.detect(window[-1]):
        cpu_thresh *= 0.9  # → 58.5
        mem_thresh *= 0.9  # → 58.5
        net_thresh *= 0.9  # → 63.0
    
    # ML FORECASTING
    if model.is_ready():
        forecast = model.predict(window)  # (5, 3)
        
        should_scale_up = any(forecast[:,0] > cpu_thresh) OR
                          any(forecast[:,1] > mem_thresh) OR
                          any(forecast[:,2] > net_thresh)
        
        should_scale_down = all(forecast[:,0] < 30) AND
                            all(forecast[:,1] < 30) AND
                            all(forecast[:,2] < 30)
    else:
        # Fallback: use last observation as naive forecast
        should_scale_up = last[0] > cpu_thresh OR last[1] > mem_thresh OR last[2] > net_thresh
    
    # Apply with cooldown
    if should_scale_up AND cooldown == 0:
        capacity = min(capacity + 1, 20)
        cooldown = 3
    elif should_scale_down AND NOT should_scale_up AND cooldown == 0:
        capacity = max(capacity - 1, 1)
        cooldown = 3
    
    return capacity
```

## 6.3 Comparative Analysis

| Aspect | Reactive | Predictive |
|--------|----------|------------|
| **Timing** | AFTER overload | BEFORE overload |
| **Trigger** | Current utilisation > threshold | Forecasted utilisation > threshold |
| **CPU threshold** | 80% | 65% (58.5% during anomaly) |
| **Scale-up confirmation** | 2 consecutive overload steps | Single forecast exceeding threshold |
| **Scale-down** | 5 consecutive low steps | All forecasted values < 30% |
| **Cooldown** | Via consecutive step counters | 3 explicit cooldown steps |
| **Anomaly awareness** | None | Lowers thresholds by 10% |
| **Spike handling** | Waits for 2 consecutive overloads | Rate-of-change acceleration fast-path |
| **Model dependency** | None | Requires trained CombinedForecaster |
| **Forecast horizon** | 0 (reactive only) | 5 steps ahead |

### Why Predictive Performs Better
1. **Lower thresholds** (65% vs 80%): Acts before overload occurs
2. **Forecast awareness**: Sees overload coming 5 steps in advance
3. **Spike fast-path**: Detects acceleration before overload manifests
4. **Anomaly safety margin**: Reduces thresholds during unusual patterns
5. **Proactive scaling**: Resources provisioned before demand spike hits

### Phase 1 Overload Reduction Results (CPU-Only)
| Pattern | Reactive Overloads | Predictive Overloads | Reduction |
|---------|-------------------|---------------------|-----------|
| Gradual | 8 | 3 | **62%** |
| Spike | 31 | 19 | **39%** |
| Periodic | 39 | 20 | **49%** |
