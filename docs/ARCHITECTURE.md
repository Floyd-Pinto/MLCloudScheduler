# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite)                     │
│                                                             │
│  Research Overview · Workload Simulation · Model Training   │
│  Findings · Metrics · Run Logs                              │
│                                                             │
│  Theme: Monochrome academic (Inter + JetBrains Mono)        │
│  Charts: Chart.js + react-chartjs-2                         │
│  HTTP:   Axios → localhost:8000/api/                        │
└─────────────────────────┬───────────────────────────────────┘
                          │  HTTP REST (JSON)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Django REST Framework                        │
│                                                             │
│  /api/simulation/    → WorkloadRun, WorkloadDataPoint       │
│  /api/scheduler/     → SchedulerRun, SchedulerAction        │
│  /api/ml/            → ModelTrainingRun                     │
│  /api/metrics/       → Aggregate views on SchedulerRun      │
│  /api/evaluation/    → EvaluationResult                     │
│                                                             │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
               ▼                      ▼
  ┌─────────────────────┐   ┌──────────────────────────────┐
  │  SQLite Database    │   │   model/ Python Package      │
  │  (db.sqlite3)       │   │                              │
  │                     │   │  workload_generator.py       │
  │  Tables:            │   │  reactive_scheduler.py       │
  │  · WorkloadRun      │   │  predictive_scheduler.py     │
  │  · WorkloadDataPoint│   │  metrics_collector.py        │
  │  · SchedulerRun     │   │  inference.py                │
  │  · SchedulerAction  │   │  lstm_model.py  (PyTorch)    │
  │  · ModelTrainingRun │   │  arima_model.py (statsmodels)│
  │  · EvaluationResult │   │  combined_model.py (ensemble)│
  └─────────────────────┘   │  train_gbr.py               │
                            │  train_lstm.py               │
                            │  train_arima.py              │
                            │  train_all.py                │
                            │                              │
                            │  saved_models/               │
                            │    gbr_model.pkl   (478 KB)  │
                            │    lstm_model.pt   (827 KB)  │
                            │    lstm_scaler.pkl            │
                            │    lstm_meta.json             │
                            │    scaler.pkl                 │
                            │    arima_meta.json            │
                            │    combined_meta.json         │
                            └──────────────────────────────┘
```

---

## Data Flow

### 1. Workload Generation
```
User clicks "Generate Workload"
  → POST /api/simulation/generate/ { pattern, steps, seed }
  → backend/simulation/services.py → model/workload_generator.py
  → numpy generates synthetic time series
  → Saved to: WorkloadRun + WorkloadDataPoint rows in SQLite
```

### 2. Model Training
```
User clicks "Train All Models"
  → POST /api/ml/train/ { model_type: "all" }
  → backend/ml_model/services.py trains sequentially:
      1. GBR  → model/train_gbr.py  → saved_models/gbr_model.pkl
      2. LSTM → model/train_lstm.py → saved_models/lstm_model.pt
      3. ARIMA→ model/train_arima.py→ saved_models/arima_meta.json
      4. Combined → model/combined_model.py → saved_models/combined_meta.json
  → Each training run recorded: ModelTrainingRun (r2, rmse, mae, status)
```

### 3. Scheduler Comparison
```
User clicks "Run Comparison"
  → POST /api/scheduler/compare/ { pattern, steps, seed }
  → backend/scheduler/services.py runs BOTH schedulers on same workload:

      Reactive Scheduler (baseline):
        for each step:
          if actual_cpu > 70% → scale_up
          if actual_cpu < 30% → scale_down (after cooldown)
          else → hold

      Predictive Scheduler (proposed):
        for each step:
          predicted = inference.predict(history, model_type="gbr")
          if predicted > 55% → scale_up    ← lower threshold = proactive
          if predicted < 25% → scale_down
          else → hold

  → Both runs saved: SchedulerRun + SchedulerAction rows
  → Response: { reactive: {...}, predictive: {...} }
```

### 4. Model Comparison
```
User clicks "Evaluate Models" on Findings page
  → POST /api/ml/compare-models/ { pattern, steps, seed }
  → backend/ml_model/services.py:
      1. Generates workload
      2. Each model forecasts via sliding window
      3. Computes R², RMSE, MAE per model
      4. Returns metrics + chart data (actual vs predicted arrays)
```

---

## ML Pipeline Detail

```
Training Data (4500+ synthetic time-series points)
         │
         ▼
   build_full_series() in train_all.py
   (gradual + spike + periodic + combined + varied seeds)
         │
         ├──► MinMaxScaler → GBR (200 trees, depth=5)
         │    Feature: window of 10 lagged values → predict next
         │
         ├──► LSTM (128 hidden, 2-layer, BatchNorm, 3 FC)
         │    Input: window of 20 normalised values → predict 5 ahead
         │    Training: 150 epochs, Adam, MSE, lr=0.001
         │
         ├──► ARIMA (auto-order via AIC grid search)
         │    Walk-forward validation on 300-step segment
         │
         └──► Combined Ensemble
              Weights: w_i = (1/RMSE_i) / Σ(1/RMSE_j)
              Prediction: w_lstm × LSTM_pred + w_arima × ARIMA_pred
```

### Why These Specific Models?

| Model | Rationale |
|---|---|
| **LSTM** | Captures non-linear temporal dependencies; state-of-the-art for sequence forecasting |
| **ARIMA** | Established statistical baseline; handles linear trends and seasonality explicitly |
| **Combined** | Tests whether ensemble of deep learning + statistical improves upon individual models |
| **GBR** | Fast, reliable baseline for the predictive scheduler's real-time inference path |

---

## Scheduling Logic

### Reactive Scheduler (Baseline — `model/reactive_scheduler.py`)
- **Trigger**: `actual_cpu > scale_up_threshold (70%)`
- **Cooldown**: 3 steps between scaling actions
- **Response**: Always **after** overload has occurred
- **Weakness**: 30–120 second provisioning lag = overload during lag period

### Predictive Scheduler (Proposed — `model/predictive_scheduler.py`)
- **Trigger**: `predicted_cpu > scale_up_threshold (55%)`
- **Cooldown**: 2 steps (faster response)
- **Forecast**: 5 steps ahead using GBR model via `inference.predict()`
- **Advantage**: Scales **before** overload occurs (proactive)
- **Key insight**: Lower threshold (55% vs 70%) is justified because predictions allow early action

---

## Workload Patterns

| Pattern | Description | Key Parameters | Use Case |
|---|---|---|---|
| `gradual` | Linear increase + Gaussian noise | slope=0.3, noise=2.0 | Organic user growth |
| `spike` | Gaussian burst at midpoint | spike_at=100, height=60 | Flash sale, viral event |
| `periodic` | Sinusoidal daily cycle | period=50, amplitude=25 | Business-hours traffic |
| `combined` | All three mixed | trend + cycle + spike | Realistic production |

---

## Database Schema (SQLite — `backend/db.sqlite3`)

```
WorkloadRun
  id, pattern, steps, seed, label, created_at

WorkloadDataPoint
  id, run_id (FK), time_step, workload

SchedulerRun
  id, scheduler_type, pattern, steps, seed,
  overload_events, overload_rate, avg_cpu, avg_capacity,
  total_cost, scale_up_count, scale_down_count, created_at

SchedulerAction
  id, run_id (FK), time_step, workload, capacity,
  cpu_usage, action, overloaded

ModelTrainingRun
  id, model_type, status, r2, rmse, mae,
  extra_info (JSON), started_at, finished_at

EvaluationResult
  id, pattern, steps, seed,
  r_overload_events, p_overload_events,
  r_overload_rate, p_overload_rate,
  overload_reduction, cost_difference, created_at
```

---

## Key Design Decisions

### Why SQLite?
- Zero configuration — ideal for viva/demo
- Django ORM ensures standard SQL; trivial PostgreSQL migration
- Sufficient for synthetic data volumes (~10K rows per comparison)

### Why Monochrome Theme?
- Academic papers are typically grayscale; the dashboard matches this aesthetic
- Print-friendly charts with different dash patterns (not just colours)
- Professional, formal appearance for research presentation

### Why 3 Proposed Models + 1 Baseline?
- **LSTM**: Deep learning capability for non-linear patterns
- **ARIMA**: Statistical rigour and interpretability
- **Combined**: Novel contribution — tests ensemble hypothesis
- **GBR**: Internal workhorse for scheduler inference (fast, reliable)

### Why Vite + React?
- Sub-second HMR for fast UI iteration
- Component reuse across 6 pages
- Chart.js integration via react-chartjs-2
