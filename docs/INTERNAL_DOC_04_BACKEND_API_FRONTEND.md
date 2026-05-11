# MASTER INTERNAL DOCUMENTATION — Part 4
# Sections 7–9: Backend, API, & Frontend Documentation

---

# SECTION 7 — DATABASE & BACKEND DOCUMENTATION

## 7.1 SQLite Schema

### Table: WorkloadRun
| Column | Type | Constraints |
|--------|------|-------------|
| id | BigAutoField | PK |
| pattern | CharField(20) | gradual/spike/periodic/combined |
| steps | IntegerField | default=200 |
| seed | IntegerField | default=42 |
| label | CharField(120) | blank=True |
| created_at | DateTimeField | auto_now_add |

### Table: WorkloadDataPoint
| Column | Type | Constraints |
|--------|------|-------------|
| id | BigAutoField | PK |
| run_id | ForeignKey → WorkloadRun | CASCADE |
| time_step | IntegerField | — |
| workload | FloatField | CPU signal (Phase 1 compat) |
| memory_usage | FloatField | default=0.0 (Phase 2) |
| network_io | FloatField | default=0.0 (Phase 2) |

### Table: SchedulerRun
| Column | Type | Notes |
|--------|------|-------|
| id | BigAutoField | PK |
| workload_run_id | FK → WorkloadRun | nullable |
| scheduler_type | CharField(20) | reactive/predictive |
| pattern | CharField(20) | — |
| steps, seed | IntegerField | — |
| overload_events | IntegerField | Total overload count |
| overload_rate | FloatField | % of steps overloaded |
| avg_cpu, avg_memory, avg_network | FloatField | Mean utilisation |
| avg_capacity | FloatField | Mean resource units |
| total_cost | FloatField | Sum of per-step costs |
| scale_up_count, scale_down_count | IntegerField | — |
| overload_cpu_count, overload_memory_count, overload_network_count | IntegerField | Phase 2: per-resource |
| created_at | DateTimeField | auto_now_add |

### Table: SchedulerAction
| Column | Type | Notes |
|--------|------|-------|
| id | BigAutoField | PK |
| run_id | FK → SchedulerRun | CASCADE |
| time_step | IntegerField | — |
| workload | FloatField | Raw CPU signal |
| capacity | IntegerField | — |
| cpu_usage | FloatField | % |
| memory_usage | FloatField | Phase 2 |
| network_io | FloatField | Phase 2 |
| overloaded | BooleanField | Any resource overloaded |
| action | CharField(15) | scale_up/scale_down/hold |
| trigger_resource | CharField(50) | Which resource(s) triggered |

### Table: ModelTrainingRun
| Column | Type | Notes |
|--------|------|-------|
| id | BigAutoField | PK |
| model_type | CharField(32) | gbr/lstm/arima/combined |
| status | CharField(16) | running/completed/failed |
| r2, rmse, mae | FloatField | nullable |
| extra_info | JSONField | Per-resource metrics, weights |
| error_msg | TextField | — |
| started_at, finished_at | DateTimeField | — |

### Table: ModelComparisonResult
| Column | Type | Notes |
|--------|------|-------|
| id | BigAutoField | PK |
| pattern | CharField(32) | — |
| series_length | IntegerField | — |
| seed | IntegerField | — |
| gbr_r2, gbr_rmse, gbr_mae | FloatField | nullable |
| lstm_r2, lstm_rmse, lstm_mae | FloatField | nullable |
| arima_r2, arima_rmse, arima_mae | FloatField | nullable |
| combined_r2, combined_rmse, combined_mae | FloatField | nullable |
| best_model | CharField(32) | — |
| created_at | DateTimeField | auto_now_add |

### Table: EvaluationResult
| Column | Type | Notes |
|--------|------|-------|
| id | BigAutoField | PK |
| pattern, steps, seed | — | Workload config |
| r_overload_events, r_overload_rate, r_avg_cpu, r_total_cost | — | Reactive stats |
| r_scale_up, r_scale_down | IntegerField | — |
| p_overload_events, p_overload_rate, p_avg_cpu, p_total_cost | — | Predictive stats |
| p_scale_up, p_scale_down | IntegerField | — |
| overload_reduction | FloatField | % improvement |
| cost_difference | FloatField | predictive - reactive |
| reactive_run_id, predictive_run_id | IntegerField | nullable FK-like |
| created_at | DateTimeField | auto_now_add |

### Table: AnomalyLogEntry
| Column | Type | Notes |
|--------|------|-------|
| id | BigAutoField | PK |
| time_step | IntegerField | — |
| cpu_usage, memory_usage, network_io | FloatField | Utilisation % |
| is_anomaly | BooleanField | Combined flag |
| z_score_flag | BooleanField | Z-Score detector |
| iforest_flag | BooleanField | IsolationForest detector |
| pattern | CharField(32) | — |
| scheduler_type | CharField(20) | — |
| created_at | DateTimeField | auto_now_add |

## 7.2 Design Decisions

### Why SQLite?
- Zero configuration — ideal for demo and viva
- Django ORM generates standard SQL → trivial PostgreSQL migration
- Sufficient for synthetic data volumes (~10K rows per comparison)
- Single file (`db.sqlite3`) — easy to backup/reset

### Why Django ORM?
- Mature migration system for schema evolution
- Built-in admin interface (available at `/admin/`)
- DRF serializers provide clean JSON serialisation
- Service layer pattern keeps business logic out of views

### PostgreSQL Migration Readiness
- Change one line in `settings.py` DATABASES config
- Run `python manage.py migrate` on PostgreSQL
- All queries use ORM — no raw SQL

---

# SECTION 8 — REST API DOCUMENTATION

## 8.1 Simulation Endpoints

### POST /api/simulation/generate/
- **Purpose**: Generate 3-resource synthetic workload
- **Request**: `{pattern, steps, seed, label?}`
- **Internal**: `simulation/services.generate_and_save()` → `model/workload_generator.generate_multivariate()`
- **DB writes**: WorkloadRun + WorkloadDataPoint (bulk)
- **Response**: Serialised run with all datapoints
- **Frontend**: SimulationPage

### GET /api/simulation/runs/
- **Purpose**: List all workload runs (summary, no datapoints)
- **Frontend**: SimulationPage history list

### GET /api/simulation/runs/{id}/ | DELETE
- **Purpose**: Get/delete specific run with datapoints

## 8.2 ML Model Endpoints

### POST /api/ml/train/
- **Purpose**: Train one or all models
- **Request**: `{model_type: "lstm"|"arima"|"combined"|"gbr"|"all"}`
- **Internal**: `ml_model/services.trigger_training()` or `trigger_train_all()`
- **DB writes**: ModelTrainingRun per model
- **Frontend**: TrainingPage

### GET /api/ml/status/
- **Purpose**: Get readiness + latest metrics for all 4 models
- **Internal**: `ml_model/services.get_model_status()` → checks saved_models/ files
- **Frontend**: DashboardPage, TrainingPage

### POST /api/ml/compare-models/
- **Purpose**: Evaluate all models on a workload, return metrics + chart data
- **Request**: `{pattern, steps, seed}`
- **Internal**: `ml_model/services.compare_all_models()` → generates workload, runs batch inference, pulls DB metrics
- **DB writes**: ModelComparisonResult
- **Frontend**: ModelComparisonPage

### POST /api/ml/predict/ | POST /api/ml/predict-all/
- **Purpose**: Run inference on a history window
- **Internal**: `model/inference.predict()` or `predict_all()`

### GET /api/ml/history/
- **Purpose**: List all training run records
- **Internal**: `ModelTrainingRun.objects.all()`
- **Frontend**: TrainingPage

## 8.3 Scheduler Endpoints

### POST /api/scheduler/compare/
- **Purpose**: Run both schedulers on same workload, return side-by-side
- **Request**: `{pattern, steps, seed}`
- **Internal**: `scheduler/services.run_comparison()` → runs reactive then predictive with anomaly logging
- **DB writes**: 2× SchedulerRun, 2× SchedulerAction (bulk), AnomalyLogEntry (bulk)
- **Frontend**: FindingsPage

### POST /api/scheduler/reactive/ | POST /api/scheduler/predictive/
- **Purpose**: Run individual scheduler

### GET /api/scheduler/runs/ | GET /api/scheduler/runs/{id}/
- **Purpose**: List/get scheduler runs with actions
- **Frontend**: LogsPage

## 8.4 Metrics & Evaluation Endpoints

### GET /api/metrics/summary/
- **Purpose**: Aggregated KPIs (avg overload, avg CPU by scheduler type)
- **Frontend**: MetricsPage, DashboardPage

### GET /api/metrics/
- **Purpose**: List all scheduler run summaries with optional filter (?type=, ?pattern=)
- **Frontend**: MetricsPage

### POST /api/evaluation/run/
- **Purpose**: Formal evaluation persisting overload_reduction %
- **Frontend**: FindingsPage

### GET /api/evaluation/
- **Purpose**: List all saved evaluation results

### GET /api/evaluation/comparison/
- **Purpose**: Get latest comparison for a pattern (?pattern=combined)

### GET /api/anomaly/logs/
- **Purpose**: Anomaly detection log entries (last 500)
- **Frontend**: AnomalyLogPage

### GET /api/anomaly/summary/
- **Purpose**: Aggregate anomaly stats (total_checked, total_anomalies, anomaly_rate)
- **Frontend**: AnomalyLogPage

---

# SECTION 9 — FRONTEND DOCUMENTATION

## 9.1 Architecture

- **Framework**: React 19.2 + Vite 8.0
- **Routing**: react-router-dom 7.14 (BrowserRouter, 8 routes)
- **HTTP**: Axios with 5-minute timeout (for training)
- **Charts**: Chart.js 4.5 + react-chartjs-2 5.3
- **Notifications**: react-hot-toast
- **Icons**: lucide-react
- **Theme**: Monochrome academic (CSS custom properties)
  - Font: Inter (body), JetBrains Mono (code/data)
  - Palette: Dark background (#0a0a0a), light text (#e5e5e5), green accents
  - Border: #2a2a2a

## 9.2 Pages

### DashboardPage (`/`)
- **Purpose**: Research overview, model status, KPI display
- **APIs**: `ml/status`, `metrics/summary`
- **Shows**: Model readiness cards (✓/✗), overall stats, project info
- **User actions**: None (read-only overview)

### SimulationPage (`/simulation`)
- **Purpose**: Generate and visualise 3-resource workloads
- **APIs**: `simulation/generate`, `simulation/runs`
- **Charts**: WorkloadChart (CPU time series)
- **User actions**: Select pattern/steps/seed, generate, view history, live mode toggle

### TrainingPage (`/training`)
- **Purpose**: Train individual or all ML models
- **APIs**: `ml/train`, `ml/status`, `ml/history`
- **Shows**: Per-model training cards, training history table
- **User actions**: Train individual model, Train All, view history

### FindingsPage (`/findings`)
- **Purpose**: Core research findings — scheduler comparison + overload analysis
- **APIs**: `scheduler/compare`, `evaluation/run`
- **Charts**: ComparisonChart (reactive vs predictive overlay), BarCompareChart (overload breakdown)
- **Shows**: Per-resource overload breakdown, overload reduction %, cost difference
- **User actions**: Run comparison, export as PNG

### ModelComparisonPage (`/comparison`)
- **Purpose**: Per-resource model accuracy comparison
- **APIs**: `ml/compare-models`
- **Charts**: ForecastChart (actual vs predicted), RadarChart (multi-model R²)
- **Shows**: Per-resource R²/RMSE/MAE for all 4 models
- **User actions**: Run evaluation, select pattern

### MetricsPage (`/metrics`)
- **Purpose**: Aggregated reactive vs predictive statistics
- **APIs**: `metrics/summary`, `metrics/`
- **Shows**: Summary cards, per-run metrics table

### LogsPage (`/logs`)
- **Purpose**: Step-by-step scheduler action inspection
- **APIs**: `scheduler/runs`, `scheduler/runs/{id}`
- **Shows**: Run list, expandable per-step action table
- **User actions**: Click run to expand actions

### AnomalyLogPage (`/anomaly`)
- **Purpose**: Anomaly detection log viewer
- **APIs**: `anomaly/logs`, `anomaly/summary`
- **Shows**: Summary stats (total checked, anomaly rate), log entry table

## 9.3 Chart Components

| Component | File | Chart Type | Used By |
|-----------|------|-----------|---------|
| WorkloadChart | `charts/WorkloadChart.jsx` | Line | SimulationPage |
| ComparisonChart | `charts/ComparisonChart.jsx` | Dual-line overlay | FindingsPage |
| ForecastChart | `charts/ForecastChart.jsx` | Multi-line (actual vs models) | ModelComparisonPage |
| BarCompareChart | `charts/BarCompareChart.jsx` | Grouped bar | FindingsPage |
| RadarChart | `charts/RadarChart.jsx` | Radar/spider | ModelComparisonPage |

## 9.4 Design Decisions

### Why Monochrome Theme?
- Academic papers are typically grayscale
- Print-friendly charts with dash patterns (not just colours)
- Professional, formal appearance for research presentation
- Consistent with IEEE paper aesthetic

### Why Chart.js?
- Lightweight (~200KB), no heavy D3 dependency
- react-chartjs-2 provides clean React integration
- Sufficient for time-series line charts and bar comparisons
- Responsive by default

### Why Vite?
- Sub-second HMR for fast UI iteration during development
- ESM-native, no Webpack configuration overhead
- Used by React team as recommended bundler

### Dependencies (package.json)
| Package | Version | Purpose |
|---------|---------|---------|
| react | 19.2.5 | UI framework |
| react-dom | 19.2.5 | DOM rendering |
| react-router-dom | 7.14.1 | Client-side routing |
| axios | 1.15.1 | HTTP client |
| chart.js | 4.5.1 | Charting library |
| react-chartjs-2 | 5.3.1 | React chart wrappers |
| react-hot-toast | 2.6.0 | Toast notifications |
| lucide-react | 1.8.0 | Icon library |
| vite | 8.0.9 | Build tool / dev server |
