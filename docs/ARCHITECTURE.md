# Architecture Notes

## System Overview

```
┌──────────────────────────────────────────────────────┐
│                   React Frontend                     │
│  Dashboard · Simulation · Training · Comparison      │
│  Metrics · Logs                                      │
│  (Vite, Chart.js, Axios, React Router)               │
└──────────────────────┬───────────────────────────────┘
                       │  HTTP REST (JSON)
                       ▼
┌──────────────────────────────────────────────────────┐
│               Django REST Framework                  │
│                                                      │
│  /api/simulation/   WorkloadRun, WorkloadDataPoint   │
│  /api/scheduler/    SchedulerRun, SchedulerAction    │
│  /api/ml/           ModelTrainingRun                 │
│  /api/metrics/      Aggregate views                  │
│  /api/evaluation/   EvaluationResult                 │
│                                                      │
└──────────────┬───────────────────┬───────────────────┘
               │                   │
               ▼                   ▼
  ┌────────────────────┐  ┌────────────────────────┐
  │   SQLite Database  │  │    model/ Python pkg   │
  │   (db.sqlite3)     │  │                        │
  │                    │  │  workload_generator.py │
  │  WorkloadRun       │  │  reactive_scheduler.py │
  │  WorkloadDataPoint │  │  predictive_scheduler  │
  │  SchedulerRun      │  │  metrics_collector.py  │
  │  SchedulerAction   │  │  inference.py          │
  │  ModelTrainingRun  │  │  train_gbr.py          │
  │  EvaluationResult  │  │  train_lstm.py         │
  └────────────────────┘  │  saved_models/         │
                          │    gbr_model.pkl       │
                          │    scaler.pkl          │
                          └────────────────────────┘
```

---

## ML Pipeline

```
Raw Workload Series (np.ndarray)
         │
         ▼
  Sliding Window (size=10 to 15)
         │
         ├──► MinMaxScaler → GBR Predict (Trees)
         ├──► PyTorch LSTM (Deep Learning)
         └──► ARIMA (Statsmodels)
         │
         ▼
  Combined Ensemble
  (Weighted dynamically by inverse-RMSE of models on recent history)
         │
         ▼
  Predictive Scheduler (Defaulting to GBR backend)
  if predicted_cpu > scale_up_threshold  → scale_up
  if predicted_cpu < scale_down_threshold  → scale_down
  else                                   → hold
```

**Reactive baseline:** waits until actual CPU > 70% then scales. Reacts one cooldown period late.

**Predictive advantage:** knows load will spike 5 steps ahead across models (GBR, LSTM, ARIMA) and pre-allocates capacity dynamically.

---

## Workload Patterns

| Pattern | Description | Parameters |
|---|---|---|
| `gradual` | Linear increase + noise | slope=0.3, noise=2.0 |
| `spike` | Gaussian burst at midpoint | spike_at=100, height=60 |
| `periodic` | Sinusoidal daily cycle | period=50, amplitude=25 |
| `combined` | All three mixed | trend + cycle + spike |

---

## Key Design Decisions

### Why SQLite?
- Zero config for viva demo
- Django ORM → trivial migration to PostgreSQL via `DATABASE_URL`
- Schema is standard SQL throughout

### Why 4 Different ML Models?
- **GradientBoostingRegressor (GBR)**: No GPU required, trains in ~2s on CPU. Interpretable with feature importances. Excellent at rapid variance matching.
- **LSTM (PyTorch)**: Demonstrates recurrent neural network proficiency. High accuracy with massive parameter flexibility, though slower to train (~20s).
- **ARIMA**: Standard statistical baseline, handles explicit trending optimally by finding the minimal AIC parameter vector dynamically.
- **Combined Ensemble**: Fuses LSTM non-linear awareness and ARIMA trend capabilities, weighted by localized training RMSE scores, making it mathematically unbiased depending on dataset patterns.

### Why Vite + React?
- Sub-second HMR for fast UI iteration
- Component reuse across 6 pages
- Chart.js integration via `react-chartjs-2`

### Why vanilla CSS?
- No build-time dependency on Tailwind/Bootstrap
- Full control over the dark theme design system
- CSS custom properties (variables) for consistent theming
