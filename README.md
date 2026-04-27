# ML-Based Adaptive Cloud Resource Scheduling

> **Major Project — 2026** — Research dashboard demonstrating ML-based predictive cloud resource scheduling versus traditional reactive scheduling using LSTM, ARIMA, and Combined ensemble models.

---

## Research Question

> Can ML-based predictive scheduling (LSTM, ARIMA, Combined ensemble) reduce cloud resource overload events by ≥40% compared to threshold-based reactive autoscaling across gradual, spike, and periodic workload patterns?

**Result**: Yes — 39–62% overload reduction achieved across all three patterns.

---

## Key Results

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| **LSTM** | **0.9696** | 5.11 | 4.04 |
| **ARIMA** | 0.6351 | 1.85 | 0.93 |
| **Combined (LSTM+ARIMA)** | 0.7952 | 14.53 | 6.56 |

| Pattern | Reactive Overloads | Predictive Overloads | Reduction |
|---|---|---|---|
| **Gradual** | 8 | 3 | **62%** |
| **Spike** | 31 | 19 | **39%** |
| **Periodic** | 39 | 20 | **49%** |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React 19, Vite 8, Chart.js | Academic research dashboard |
| Backend | Django 5, Django REST Framework | REST API, data persistence |
| ML Models | PyTorch (LSTM), Statsmodels (ARIMA), Scikit-learn (GBR) | Workload forecasting |
| Database | SQLite | Zero-config local storage |
| Theme | Monochrome (Inter + JetBrains Mono) | Formal academic presentation |

---

## Project Structure

```
ml-cloud-scheduler/
│
├── README.md                    ← You are here
├── requirements.txt             ← Top-level Python dependencies
├── Makefile                     ← Shortcuts (make run-backend, etc.)
├── Dockerfile                   ← Docker containerisation
├── docker-compose.yml           ← Multi-container setup
├── .env                         ← Environment variables
├── workload_generator.py        ← Standalone workload generator script
│
├── model/                       ← ML PIPELINE (core research code)
│   ├── __init__.py
│   ├── workload_generator.py    ← Generate synthetic workloads (gradual/spike/periodic/combined)
│   ├── train_all.py             ← Master training script — trains all 4 models sequentially
│   ├── train_gbr.py             ← GBR training (scikit-learn GradientBoostingRegressor)
│   ├── train_lstm.py            ← LSTM training (PyTorch)
│   ├── train_arima.py           ← ARIMA training (statsmodels)
│   ├── lstm_model.py            ← LSTM architecture: LSTMForecaster class (128 hidden, 2-layer, BatchNorm)
│   ├── arima_model.py           ← ARIMA model: ARIMAForecaster class (auto-order via AIC)
│   ├── combined_model.py        ← Combined ensemble: CombinedForecaster (LSTM+ARIMA, inverse-RMSE weights)
│   ├── inference.py             ← Unified prediction interface for all model types
│   ├── evaluate.py              ← Model evaluation utilities (R², RMSE, MAE)
│   ├── reactive_scheduler.py    ← Reactive scheduler: threshold-based (CPU > 70% → scale up)
│   ├── predictive_scheduler.py  ← Predictive scheduler: ML forecast → proactive scaling (threshold 55%)
│   ├── metrics_collector.py     ← Collect per-step metrics (CPU, capacity, overload, cost)
│   │
│   ├── saved_models/            ← TRAINED MODEL ARTIFACTS (binary files)
│   │   ├── gbr_model.pkl        ← Serialised GBR model (joblib, ~478 KB)
│   │   ├── scaler.pkl           ← MinMaxScaler for GBR features (joblib)
│   │   ├── lstm_model.pt        ← PyTorch LSTM state_dict (~827 KB)
│   │   ├── lstm_scaler.pkl      ← MinMaxScaler for LSTM input normalisation
│   │   ├── lstm_meta.json       ← LSTM hyperparameters (hidden_size, window_size)
│   │   ├── arima_meta.json      ← ARIMA order + seasonal info
│   │   └── combined_meta.json   ← Combined weights (w_lstm, w_arima)
│   │
│   ├── data/                    ← Training data cache (generated during training)
│   └── scripts/                 ← Utility scripts
│
├── backend/                     ← DJANGO REST API
│   ├── manage.py                ← Django management entry point
│   ├── requirements.txt         ← Backend Python dependencies
│   ├── db.sqlite3               ← SQLite database (all training records, scheduler runs, metrics)
│   │
│   ├── config/                  ← Django project configuration
│   │   ├── settings.py          ← Django settings (CORS, REST framework, apps, database)
│   │   ├── urls.py              ← Root URL routing → /api/simulation/, /api/ml/, etc.
│   │   └── wsgi.py              ← WSGI application entry point
│   │
│   ├── simulation/              ← Workload Simulation app
│   │   ├── models.py            ← WorkloadRun, WorkloadDataPoint models
│   │   ├── serializers.py       ← DRF serializers
│   │   ├── services.py          ← generate_workload() — calls model/workload_generator.py
│   │   ├── views.py             ← POST /generate/, GET/DELETE /runs/
│   │   └── urls.py              ← Route definitions
│   │
│   ├── ml_model/                ← ML Training & Inference app
│   │   ├── models.py            ← ModelTrainingRun model (stores R², RMSE, MAE per training)
│   │   ├── serializers.py       ← DRF serializers
│   │   ├── services.py          ← train_model(), predict(), compare_all_models()
│   │   ├── views.py             ← POST /train/, GET /status/, POST /predict/, POST /compare-models/
│   │   └── urls.py              ← Route definitions
│   │
│   ├── scheduler/               ← Scheduler Comparison app
│   │   ├── models.py            ← SchedulerRun, SchedulerAction models
│   │   ├── serializers.py       ← DRF serializers
│   │   ├── services.py          ← run_reactive(), run_predictive(), compare_schedulers()
│   │   ├── views.py             ← POST /reactive/, POST /predictive/, POST /compare/
│   │   └── urls.py              ← Route definitions
│   │
│   ├── metrics/                 ← Metrics Aggregation app
│   │   ├── models.py            ← (uses SchedulerRun from scheduler app)
│   │   ├── services.py          ← get_summary(), get_filtered_list()
│   │   ├── views.py             ← GET /metrics/, GET /metrics/summary/
│   │   └── urls.py              ← Route definitions
│   │
│   └── evaluation/              ← Evaluation & Comparison app
│       ├── models.py            ← EvaluationResult model
│       ├── services.py          ← run_full_evaluation()
│       ├── views.py             ← POST /evaluation/run/, GET /evaluation/
│       └── urls.py              ← Route definitions
│
├── frontend/                    ← REACT DASHBOARD
│   ├── package.json             ← Node dependencies (react, vite, chart.js, axios)
│   ├── vite.config.js           ← Vite configuration (proxy to Django backend)
│   ├── index.html               ← HTML entry point
│   │
│   └── src/
│       ├── main.jsx             ← React entry point
│       ├── App.jsx              ← Router: 6 pages (/, /simulation, /training, /findings, /metrics, /logs)
│       │
│       ├── styles/
│       │   └── global.css       ← Design system (monochrome academic theme, CSS variables)
│       │
│       ├── components/
│       │   └── Sidebar.jsx      ← Navigation: Research / Experiment / Results sections
│       │
│       ├── pages/
│       │   ├── DashboardPage.jsx      ← Research Overview (hypothesis, model status, key findings)
│       │   ├── SimulationPage.jsx     ← Workload Simulation (generate gradual/spike/periodic patterns)
│       │   ├── TrainingPage.jsx       ← Model Training (LSTM, ARIMA, Combined — train & view metrics)
│       │   ├── FindingsPage.jsx       ← Findings (scheduler comparison + model accuracy tabs)
│       │   ├── MetricsPage.jsx        ← Metrics (aggregated reactive vs predictive stats)
│       │   └── LogsPage.jsx           ← Run Logs (step-by-step scheduler actions)
│       │
│       ├── charts/
│       │   ├── WorkloadChart.jsx      ← Line chart for workload patterns
│       │   ├── ForecastChart.jsx      ← Multi-line forecast (actual vs LSTM/ARIMA/Combined)
│       │   ├── ComparisonChart.jsx    ← Dual-line reactive vs predictive capacity over time
│       │   ├── BarCompareChart.jsx    ← Bar chart for metric comparison
│       │   └── RadarChart.jsx         ← Radar chart for multi-model comparison
│       │
│       └── services/
│           └── api.js                 ← Axios HTTP client (mlAPI, schedulerAPI, simulationAPI, etc.)
│
├── data/                        ← Static data files (if any)
├── outputs/                     ← Generated outputs (plots, exports)
│
└── docs/                        ← DOCUMENTATION
    ├── SETUP.md                 ← Installation & setup guide
    ├── ARCHITECTURE.md          ← System architecture & design decisions
    ├── API.md                   ← Full REST API reference
    └── MODELS.md                ← ML model details & training pipeline (NEW)
```

---

## Quick Start

### Prerequisites
- Python 3.11+ (3.14 confirmed working)
- Node.js 20+ (install via nvm if needed)

### 1. Set up Python environment

```bash
git clone <repo-url>
cd ml-cloud-scheduler
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn joblib statsmodels matplotlib
```

### 2. Train the models

```bash
python model/train_all.py
# Trains: GBR → LSTM → ARIMA → Combined (~2-3 minutes total)
# Outputs saved to: model/saved_models/
```

### 3. Start the backend

```bash
cd backend
python manage.py migrate
python manage.py runserver
# → http://localhost:8000
```

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### 5. Demo workflow

1. Open http://localhost:5173
2. **Overview** → See model status and key findings
3. **Workload Simulation** → Generate a workload (pattern: combined, steps: 200)
4. **Model Training** → Click "Train All Models" (~2 minutes)
5. **Findings** → Run scheduler comparison → see overload reduction %
6. **Metrics** → View aggregated stats
7. **Run Logs** → Browse step-by-step scheduler decisions

---

## Where Are Models Stored?

| Model | Training Script | Saved Artifact | Size |
|---|---|---|---|
| **GBR** | `model/train_gbr.py` | `model/saved_models/gbr_model.pkl` | ~478 KB |
| **GBR Scaler** | (same) | `model/saved_models/scaler.pkl` | ~1 KB |
| **LSTM** | `model/train_lstm.py` | `model/saved_models/lstm_model.pt` | ~827 KB |
| **LSTM Scaler** | (same) | `model/saved_models/lstm_scaler.pkl` | ~1 KB |
| **LSTM Meta** | (same) | `model/saved_models/lstm_meta.json` | JSON config |
| **ARIMA** | `model/train_arima.py` | `model/saved_models/arima_meta.json` | JSON (order, AIC) |
| **Combined** | `model/combined_model.py` | `model/saved_models/combined_meta.json` | JSON (w_lstm, w_arima) |

Training records (R², RMSE, MAE, timestamps) are stored in: `backend/db.sqlite3` → `ModelTrainingRun` table.

---

## ML Model Architectures

### 1. LSTM (Long Short-Term Memory)
- **Architecture**: 2-layer LSTM → BatchNorm → 3 Fully Connected layers → single output
- **Hidden size**: 128 units
- **Input window**: 20 time steps
- **Forecast horizon**: 5 steps ahead
- **Training**: 150 epochs, Adam optimiser, MSE loss, lr=0.001
- **R²**: 0.9696

### 2. ARIMA (Auto-Regressive Integrated Moving Average)
- **Order selection**: Grid search over p∈[0,5], d∈[0,2], q∈[0,5] minimising AIC
- **Implementation**: statsmodels SARIMAX
- **Validation**: Walk-forward on 300-step segment
- **R²**: 0.6351

### 3. Combined Ensemble (LSTM + ARIMA)
- **Weighting**: Inverse-RMSE — the model with lower RMSE gets higher weight
- **Formula**: `w_i = (1/RMSE_i) / Σ(1/RMSE_j)` for i ∈ {lstm, arima}
- **Current weights**: w_lstm=0.434, w_arima=0.566
- **R²**: 0.7952

### 4. GBR (Gradient Boosting Regressor) — Baseline
- **Architecture**: 200 trees, max_depth=5
- **Training**: scikit-learn, <2 seconds
- **Role**: Internal baseline for the predictive scheduler; not a proposed model
- **R²**: 0.9709

---

## API Quick Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/simulation/generate/` | Generate synthetic workload |
| GET | `/api/simulation/runs/` | List workload runs |
| POST | `/api/ml/train/` | Train a model (`model_type`: lstm/arima/combined/gbr/all) |
| GET | `/api/ml/status/` | Model readiness + metrics |
| POST | `/api/ml/compare-models/` | Evaluate all models on a workload |
| POST | `/api/scheduler/compare/` | Run reactive vs predictive |
| GET | `/api/metrics/summary/` | Aggregate performance stats |
| GET | `/api/scheduler/runs/` | List all scheduler runs |

See [docs/API.md](docs/API.md) for full API documentation.

---

## Documentation

| Document | Path | Content |
|---|---|---|
| **README** | `README.md` | Project overview, structure, quick start |
| **Setup Guide** | `docs/SETUP.md` | Step-by-step installation |
| **Architecture** | `docs/ARCHITECTURE.md` | System design, ML pipeline, design decisions |
| **API Reference** | `docs/API.md` | All REST endpoints with request/response examples |
| **Model Details** | `docs/MODELS.md` | ML model architectures, hyperparameters, training pipeline |
