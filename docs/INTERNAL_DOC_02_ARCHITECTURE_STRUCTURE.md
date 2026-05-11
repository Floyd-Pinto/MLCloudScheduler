# MASTER INTERNAL DOCUMENTATION — Part 2
# Sections 3–4: Architecture & Project Structure

---

# SECTION 3 — CURRENT SYSTEM ARCHITECTURE

## 3.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   React Frontend (Vite 8)                         │
│  8 Pages: Dashboard, Simulation, Training, Findings,             │
│           ModelComparison, Metrics, Logs, AnomalyLog             │
│  Theme: Monochrome academic (Inter + JetBrains Mono)             │
│  Charts: Chart.js 4 + react-chartjs-2                            │
│  HTTP: Axios → localhost:8000/api/                               │
└──────────────────────┬───────────────────────────────────────────┘
                       │ HTTP REST (JSON)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              Django 6 + Django REST Framework 3.17               │
│                                                                  │
│  Apps:                                                           │
│    simulation/  → WorkloadRun, WorkloadDataPoint                 │
│    scheduler/   → SchedulerRun, SchedulerAction                  │
│    ml_model/    → ModelTrainingRun, ModelComparisonResult         │
│    metrics/     → Aggregate views on SchedulerRun                │
│    evaluation/  → EvaluationResult                               │
│    anomaly/     → AnomalyLogEntry                                │
│                                                                  │
│  Each app: models.py → serializers.py → services.py → views.py  │
└──────────┬──────────────────────────┬────────────────────────────┘
           │                          │
           ▼                          ▼
┌────────────────────┐    ┌─────────────────────────────────────┐
│  SQLite Database   │    │     model/ Python Package           │
│  (db.sqlite3)      │    │                                     │
│                    │    │  workload_generator.py               │
│  7 Tables:         │    │  lstm_model.py      (PyTorch)       │
│  · WorkloadRun     │    │  arima_model.py     (statsmodels)   │
│  · WorkloadDataPoint│   │  combined_model.py  (ensemble)      │
│  · SchedulerRun    │    │  train_gbr.py       (sklearn)       │
│  · SchedulerAction │    │  anomaly_detector.py (IsolationForest)│
│  · ModelTrainingRun│    │  inference.py        (unified API)  │
│  · ModelComparison │    │  reactive_scheduler.py               │
│  · EvaluationResult│    │  predictive_scheduler.py             │
│  · AnomalyLogEntry │    │  metrics_collector.py                │
│                    │    │  evaluate.py                         │
│                    │    │  train_all.py                        │
│                    │    │                                     │
│                    │    │  saved_models/ (12 artifacts)        │
└────────────────────┘    └─────────────────────────────────────┘
```

## 3.2 Docker Architecture

```
docker-compose.yml (3 services)
│
├── trainer (Dockerfile.backend)
│   └── python model/train_all.py
│       → Writes to volume: model_artifacts
│       → Exits after training (~3-5 min)
│
├── backend (Dockerfile.backend)
│   ├── depends_on: trainer (service_completed_successfully)
│   ├── Reads from volume: model_artifacts
│   ├── Writes to volume: db_data (SQLite)
│   └── python manage.py migrate && runserver 0.0.0.0:8000
│
└── frontend (Dockerfile.frontend)
    ├── depends_on: backend
    ├── Multi-stage build: node:20-slim → build → serve
    ├── ARG VITE_API_URL=http://localhost:8000/api
    └── serve -s dist -l 3000

Volumes:
  model_artifacts → shared between trainer and backend
  db_data → persists SQLite database
```

## 3.3 Execution Flows

### Workload Generation Flow
```
User (SimulationPage) → POST /api/simulation/generate/
  → simulation/views.py → simulation/services.generate_and_save()
    → model/workload_generator.generate_multivariate(pattern, steps, seed)
      → generate_{pattern}() → CPU signal (1-D)
      → derive_memory_usage(cpu, rng) → Memory signal
      → derive_network_io(cpu, rng) → Network signal
      → np.column_stack([cpu, mem, net]) → (steps, 3)
    → WorkloadRun.objects.create()
    → WorkloadDataPoint.objects.bulk_create()
  → Response: serialised WorkloadRun + datapoints
```

### Training Flow
```
User (TrainingPage) → POST /api/ml/train/ {model_type: "all"}
  → ml_model/views.py → ml_model/services.trigger_train_all()
    → For each model in [gbr, lstm, arima, combined]:
      → ModelTrainingRun.objects.create(status="running")
      → _run_training(model_type):
          gbr:      model/train_gbr.train()
          lstm:     model/train_lstm.train()
          arima:    model/train_arima.train()
          combined: CombinedForecaster.fit()
      → Record metrics (r2, rmse, mae, extra_info)
      → inference.invalidate_cache()
      → ModelTrainingRun.save(status="completed")
```

### Scheduler Comparison Flow
```
User (FindingsPage) → POST /api/scheduler/compare/
  → scheduler/views.py → scheduler/services.run_comparison()
    → generate_multivariate(pattern, steps, seed) → (steps, 3)
    → _run_and_save(workload, ReactiveScheduler, ...)
      → For each step:
        → Derive cpu_pct, mem_pct, net_pct from raw + capacity
        → scheduler.decide(load, cpu_pct, mem_pct, net_pct)
        → collector.record(...)
      → SchedulerRun.objects.create(summary stats)
      → SchedulerAction.objects.bulk_create(per-step actions)
    → _run_and_save(workload, PredictiveScheduler, ...)
      → For each step:
        → Load AnomalyDetector, check z-score + iforest
        → scheduler.observe(load, cpu_pct, mem_pct, net_pct)
        → scheduler.decide() → uses CombinedForecaster.predict()
        → collector.record(...)
      → AnomalyLogEntry.objects.bulk_create(anomaly entries)
    → Return {reactive_id, predictive_id}
```

### Inference Flow
```
inference.predict(model_type, window)
  → _get_model(model_type):  # singleton cache
      gbr:      joblib.load(gbr_model.pkl + scaler.pkl)
      lstm:     LSTMForecaster().load()
      arima:    ARIMAForecaster().load()
      combined: CombinedForecaster().load()
  → Normalise input to (20, 3)
  → Model-specific prediction:
      gbr: scaler.transform → flatten → model.predict → reshape → inverse_transform
      lstm: scaler.transform → torch tensor → model(x) → inverse_transform
      arima: per-channel ARIMA.forecast(steps=5) → stack
      combined: w_lstm[r]*lstm_pred[r] + w_arima[r]*arima_pred[r]
  → Return {"cpu": [5 values], "memory": [5], "network": [5]}
```

---

# SECTION 4 — COMPLETE PROJECT STRUCTURE

## 4.1 Root Level

| File/Dir | Purpose |
|----------|---------|
| `model/` | **ML pipeline** — all models, training, inference, schedulers |
| `backend/` | **Django REST API** — 6 apps, ORM, services, views |
| `frontend/` | **React dashboard** — 8 pages, charts, API service |
| `data/` | Simulation output CSVs (8 files, reactive+predictive per pattern) |
| `outputs/` | Matplotlib PNGs from standalone evaluations |
| `docs/` | Technical documentation (ARCHITECTURE, MODELS, API, SETUP) |
| `docker-compose.yml` | 3-service orchestration |
| `Dockerfile.backend` | Python 3.12-slim, Django server |
| `Dockerfile.frontend` | Node 20, multi-stage build + serve |
| `Dockerfile.legacy` | Deprecated single-container setup |
| `Makefile` | 10 build/run targets |
| `README.md` | Project overview, results, quick start |
| `RUNNING.md` | Detailed setup and execution guide |
| `requirements.txt` | Root-level minimal deps (not primary; use backend/requirements.txt) |
| `.env` | Django secret key, debug flag, allowed hosts |
| `workload_generator.py` | **Deprecated** root-level copy (kept for backward compat) |

## 4.2 model/ — ML Pipeline

| File | Purpose | Key Classes/Functions | I/O |
|------|---------|----------------------|-----|
| `__init__.py` | Package marker | — | — |
| `workload_generator.py` | Synthetic 3-resource workload generation | `generate()`, `generate_multivariate()`, `derive_memory_usage()`, `derive_network_io()` | → (steps,3) ndarray |
| `lstm_model.py` | Multi-resource LSTM forecaster | `LSTMForecaster` (fit, predict, save, load), `get_lstm()` singleton | (20,3)→(5,3) |
| `arima_model.py` | Per-channel ARIMA forecaster | `ARIMAForecaster` (fit, predict, save, load), `get_arima()` | (n,3)→(steps,3) |
| `combined_model.py` | Per-resource weighted ensemble | `CombinedForecaster` (fit, predict, save, load), `get_combined()` | (20,3)→(5,3) |
| `train_gbr.py` | Multi-output GBR training | `train()`, `build_dataset()` | → gbr_model.pkl, scaler.pkl |
| `train_lstm.py` | LSTM training script | `train()`, `build_training_data()` | → lstm_model.pt, lstm_scaler.pkl |
| `train_arima.py` | ARIMA training script | `train()` | → arima_meta.json, arima_history.pkl |
| `train_all.py` | Master pipeline (5 components) | `train_all()` | → all saved_models/ artifacts |
| `anomaly_detector.py` | Dual-strategy anomaly detection | `AnomalyDetector` (fit, detect, detect_batch, save, load) | observation→bool |
| `inference.py` | Unified prediction interface | `predict()`, `predict_all()`, `model_is_ready()`, `invalidate_cache()` | window→dict |
| `reactive_scheduler.py` | Multi-resource reactive scheduler | `ReactiveScheduler` (decide, reset) | load+pcts→capacity |
| `predictive_scheduler.py` | ML-based predictive scheduler | `PredictiveScheduler` (observe, decide, reset) | history→capacity |
| `metrics_collector.py` | Per-step simulation recording | `MetricsCollector` (record, summary, to_dataframe), `StepRecord` | steps→summary dict |
| `evaluate.py` | CLI scheduler comparison | `run_simulation()`, `compare_schedulers()` | pattern→comparison dict |

### model/saved_models/ — 12 Trained Artifacts

| File | Size | Source |
|------|------|--------|
| `gbr_model.pkl` | 13.3 MB | MultiOutputRegressor(GBR×15) |
| `scaler.pkl` | 799 B | MinMaxScaler (3-feature) |
| `gbr_meta.json` | 67 B | {n_features, window_size, forecast_horizon} |
| `lstm_model.pt` | 846 KB | PyTorch state_dict |
| `lstm_scaler.pkl` | 799 B | MinMaxScaler (3-feature) |
| `lstm_meta.json` | 108 B | Architecture config |
| `arima_meta.json` | 181 B | Per-channel orders + n_features |
| `arima_history.pkl` | 7.4 KB | Training history for prediction |
| `combined_meta.json` | 202 B | Per-resource LSTM/ARIMA weights |
| `anomaly_iforest.pkl` | 3.1 MB | Fitted IsolationForest (200 trees) |
| `anomaly_meta.json` | 73 B | z_threshold, rolling_window, contamination |
| `training_summary.json` | 2.1 KB | All model metrics from last training |

## 4.3 backend/ — Django REST API

### App Architecture Pattern
Each app follows: `models.py → serializers.py → services.py → views.py → urls.py`

| App | ORM Models | Service Functions | Views/Endpoints |
|-----|-----------|-------------------|-----------------|
| **simulation/** | WorkloadRun, WorkloadDataPoint | generate_and_save() | generate, listRuns, getRun, deleteRun |
| **scheduler/** | SchedulerRun, SchedulerAction | run_reactive(), run_predictive(), run_comparison() | reactive, predictive, compare, listRuns, getRun |
| **ml_model/** | ModelTrainingRun, ModelComparisonResult | trigger_training(), compare_all_models(), run_inference() | train, status, predict, predict-all, compare-models, history |
| **metrics/** | (uses SchedulerRun) | — | list (filtered), summary |
| **evaluation/** | EvaluationResult | run_full_evaluation(), get_latest_comparison() | run, list, comparison |
| **anomaly/** | AnomalyLogEntry | — | logs, summary |
| **config/** | — | — | Root URL routing, settings |

## 4.4 frontend/src/ — React Dashboard

| File/Dir | Purpose |
|----------|---------|
| `App.jsx` | BrowserRouter with 8 routes + Sidebar + Toaster |
| `main.jsx` | React 19 entry point |
| `components/Sidebar.jsx` | Navigation sidebar with 3 sections (Research, Experiment, Results) |
| `services/api.js` | Axios API client with all endpoint wrappers |
| `styles/global.css` | Monochrome academic theme (CSS variables, Inter, JetBrains Mono) |
| `charts/WorkloadChart.jsx` | Line chart for workload time series |
| `charts/ComparisonChart.jsx` | Dual-line reactive vs predictive overlay |
| `charts/ForecastChart.jsx` | Actual vs predicted forecast lines |
| `charts/BarCompareChart.jsx` | Bar chart for metric comparisons |
| `charts/RadarChart.jsx` | Radar chart for multi-model comparison |

### Page Responsibilities

| Page | Route | APIs Used | Purpose |
|------|-------|-----------|---------|
| DashboardPage | `/` | ml/status, metrics/summary | Overview KPIs, model status cards |
| SimulationPage | `/simulation` | simulation/generate, runs | Generate + view workloads, live mode |
| TrainingPage | `/training` | ml/train, ml/status, ml/history | Train models, view training history |
| FindingsPage | `/findings` | scheduler/compare, evaluation/run | Scheduler comparison, overload breakdown, PNG export |
| ModelComparisonPage | `/comparison` | ml/compare-models | Per-resource model metrics + forecast charts |
| MetricsPage | `/metrics` | metrics/summary, metrics/ | Aggregated reactive vs predictive stats |
| LogsPage | `/logs` | scheduler/runs, scheduler/runs/{id} | Step-by-step scheduler action viewer |
| AnomalyLogPage | `/anomaly` | anomaly/logs, anomaly/summary | Anomaly detection log table + stats |
