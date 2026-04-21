# ML-Based Adaptive Cloud Resource Scheduling

> **Final Year Major Project** — Full-stack web application demonstrating ML-based predictive cloud resource scheduling versus traditional reactive scheduling.

---

## Project Overview

This system simulates cloud workload environments and compares two scheduling strategies:

| Scheduler | Strategy | Description |
|---|---|---|
| **Reactive** | Threshold-based | Scales resources *after* CPU exceeds a threshold (standard autoscaler behaviour) |
| **Predictive** | ML-based | Uses GBR, PyTorch LSTM, ARIMA, or a Combined ensemble to forecast load and scales *before* overload |

The frontend dashboard visualizes workload patterns, CPU utilization, capacity decisions, overload events, cost, and comparative ML metrics — making the advantage of predictive scheduling clearly demonstrable.

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Frontend | React + Vite | Fast dev server, component-based UI |
| Backend | Django + Django REST Framework | Robust, batteries-included Python API framework |
| Database | SQLite (dev) | Zero-config, PostgreSQL-compatible schema for easy upgrade |
| ML Model | GBR, PyTorch LSTM, ARIMA, Combined Ensemble | Deep learning, statistical, and hybrid models compared side-by-side |
| Charts | Chart.js + react-chartjs-2 | Lightweight, well-documented charting |
| CSS | Vanilla CSS (dark theme) | No framework dependency, full control |

---

## Folder Structure

```
ml-cloud-scheduler/
├── frontend/          # React + Vite app
│   ├── src/
│   │   ├── pages/     # Dashboard, Simulation, Training, Comparison, Metrics, Logs
│   │   ├── charts/    # Chart.js wrappers
│   │   ├── components/# Sidebar, reusable UI
│   │   ├── services/  # Axios API layer
│   │   └── styles/    # Global CSS (dark theme)
│   └── package.json
│
├── backend/           # Django project
│   ├── config/        # Settings, URLs, WSGI
│   ├── simulation/    # Workload generation API
│   ├── scheduler/     # Reactive + predictive scheduler API
│   ├── ml_model/      # Training trigger + inference API
│   ├── metrics/       # Metrics retrieval API
│   ├── evaluation/    # Comparison + evaluation API
│   └── manage.py
│
├── model/             # ML code and artifacts
│   ├── workload_generator.py
│   ├── reactive_scheduler.py
│   ├── predictive_scheduler.py
│   ├── metrics_collector.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── train_all.py   # ← run this to (re)train all 4 models
│   ├── lstm_model.py  # PyTorch LSTM
│   ├── arima_model.py # Statsmodels ARIMA
│   ├── combined_model.py # Inverse-RMSE weighted ensemble
│   └── saved_models/
│       ├── gbr_model.pkl
│       └── scaler.pkl
│
└── docs/
    ├── SETUP.md
    ├── ARCHITECTURE.md
    └── API.md
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+ (installed via nvm if needed)

### 1. Clone & set up Python env

```bash
git clone <repo-url>
cd ml-cloud-scheduler
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn joblib statsmodels matplotlib
```

### 2. Train the models (first time)

```bash
python model/train_all.py
# Output: trains and saves GBR, PyTorch LSTM, ARIMA, and Combined Hybrid models
```

### 3. Start the backend

```bash
cd backend
python manage.py migrate    # already done — db.sqlite3 is committed
python manage.py runserver
# → http://localhost:8000
```

### 4. Start the frontend

```bash
# In a new terminal
export NVM_DIR="$HOME/.nvm" && source "$NVM_DIR/nvm.sh"
cd frontend
npm install    # if not already done
npm run dev
# → http://localhost:5173
```

### 5. Seed data (optional demo shortcut)

Go to the UI:
1. **Simulation** → Generate a workload (pattern: combined, steps: 200)
2. **Training** → Click "Start Training"
3. **Comparison** → Run Comparison → See results

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/simulation/generate/` | Generate synthetic workload |
| GET | `/api/simulation/runs/` | List workload runs |
| GET | `/api/simulation/runs/{id}/` | Get run + datapoints |
| POST | `/api/scheduler/reactive/` | Run reactive scheduler |
| POST | `/api/scheduler/predictive/` | Run predictive scheduler |
| POST | `/api/scheduler/compare/` | Run both + return side-by-side |
| GET | `/api/scheduler/runs/` | List all scheduler runs |
| GET | `/api/scheduler/runs/{id}/` | Get run + step actions |
| POST | `/api/ml/train/` | Trigger model training or train "all" |
| GET | `/api/ml/status/` | Readiness + metrics for all models |
| POST | `/api/ml/predict/` | Inference on history window |
| POST | `/api/ml/predict-all/` | Inference across all models simultaneously |
| POST/GET | `/api/ml/compare-models/` | Evaluate all 4 models on identical workloads + retrieve charts |
| GET | `/api/ml/history/` | Training run history |
| GET | `/api/metrics/` | All scheduler run summaries |
| GET | `/api/metrics/summary/` | Aggregate KPIs |
| POST | `/api/evaluation/run/` | Full evaluation + save |
| GET | `/api/evaluation/` | List evaluations |
| GET | `/api/evaluation/comparison/` | Latest comparison result |

---

## Database Choice: SQLite

SQLite was chosen for local development because:
- Zero configuration — works out of the box
- Django ORM ensures all queries are standard SQL, making PostgreSQL migration trivial (`DATABASE_URL` + `psycopg2`)
- Sufficient for the volume of data generated (thousands of rows per run)

To switch to PostgreSQL: change `DATABASES` in `backend/config/settings.py`.

---

## ML Model Architectures

The system features four distinct forecasting engines configured dynamically via API:
1. **GradientBoostingRegressor** - 200 tree ensemble, ~2-second training, fast scaling logic.
2. **PyTorch LSTM** - 2-Layer Recurrent Neural Net with Adam optimization capturing long-term non-linearities.
3. **ARIMA** - Statsmodels statistical baseline utilizing AIC-based order grid searching.
4. **Combined Hybrid Ensemble** - Weights the LSTM and ARIMA engines dynamically scaled by their inverse-RMSE accuracy.

---


