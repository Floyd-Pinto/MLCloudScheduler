# RUNNING.md — Setup & Execution Guide

## ML-Based Adaptive Cloud Resource Scheduling (Phase 2)

---

## Prerequisites

| Component | Version |
|-----------|---------|
| Python    | 3.11+   |
| Node.js   | 20+     |
| Docker    | 24+ (optional) |

---

## Quick Start (Manual)

### 1. Clone & Install

```bash
git clone https://github.com/Floyd-Pinto/MLCloudScheduler.git
cd MLCloudScheduler

# Python virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r backend/requirements.txt

# Install Node dependencies
cd frontend && npm install && cd ..
```

Or use the shortcut:
```bash
make install
```

### 2. Train All Models

```bash
make train
# or:
source .venv/bin/activate && MPLBACKEND=Agg python model/train_all.py
```

This trains all 5 components sequentially:
1. **GBR** (baseline) — MultiOutput GradientBoosting
2. **LSTM** — Multi-resource, 2-layer with BatchNorm
3. **ARIMA** — Per-channel AIC grid search
4. **Combined** — Per-resource inverse-RMSE ensemble
5. **AnomalyDetector** — IsolationForest + Rolling Z-Score

Outputs saved to `model/saved_models/`.
Training takes approximately 3-5 minutes on CPU.

### 3. Run Database Migrations

```bash
make migrate
# or:
cd backend && python manage.py makemigrations && python manage.py migrate
```

### 4. Start Development Servers

```bash
# Start both backend + frontend:
make dev

# Or separately:
make backend    # Django on http://localhost:8000
make frontend   # Vite on http://localhost:5173
```

### 5. Verify

Open **http://localhost:5173** in your browser.

Smoke-test all API endpoints:
```bash
make test-api
```

---

## Quick Start (Docker)

```bash
# Build and run all 3 services (trainer → backend → frontend):
make docker-up
# or:
docker compose up --build
```

| Service  | Port | Description |
|----------|------|-------------|
| trainer  | —    | Runs `train_all.py` once, then exits |
| backend  | 8000 | Django REST API |
| frontend | 3000 | React/Vite (production build) |

Backend waits for trainer to complete before starting.

Stop with:
```bash
make docker-down
```

---

## Project Structure

```
MLCloudScheduler/
├── model/                          # ML pipeline
│   ├── workload_generator.py       # 3-resource signal generation
│   ├── lstm_model.py               # Multi-input/output LSTM
│   ├── arima_model.py              # Per-channel ARIMA
│   ├── combined_model.py           # Per-resource ensemble
│   ├── train_gbr.py                # MultiOutput GBR trainer
│   ├── train_lstm.py               # LSTM trainer
│   ├── train_arima.py              # ARIMA trainer
│   ├── train_all.py                # Master training pipeline
│   ├── anomaly_detector.py         # IsolationForest + Z-Score
│   ├── inference.py                # Unified prediction interface
│   ├── reactive_scheduler.py       # Multi-resource reactive
│   ├── predictive_scheduler.py     # ML-based predictive
│   ├── metrics_collector.py        # Per-step multi-resource metrics
│   ├── evaluate.py                 # Scheduler comparison runner
│   └── saved_models/               # Trained model artifacts
├── backend/                        # Django REST API
│   ├── config/                     # Django settings & URLs
│   ├── simulation/                 # Workload generation endpoints
│   ├── scheduler/                  # Scheduler execution endpoints
│   ├── ml_model/                   # Training & inference endpoints
│   ├── metrics/                    # Aggregated metrics
│   ├── evaluation/                 # Evaluation persistence
│   └── anomaly/                    # Anomaly detection logs
├── frontend/                       # React/Vite dashboard
│   └── src/
│       ├── pages/                  # 8 page components
│       ├── charts/                 # Chart.js chart components
│       ├── components/             # Sidebar
│       ├── services/               # API client (axios)
│       └── styles/                 # global.css
├── docker-compose.yml              # 3-service Docker setup
├── Dockerfile.backend              # Django container
├── Dockerfile.frontend             # React build + serve
└── Makefile                        # All development shortcuts
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MPLBACKEND` | `Agg` | Matplotlib backend (headless) |
| `VITE_API_URL` | `http://localhost:8000/api` | Backend API URL for frontend |
| `DJANGO_DEBUG` | `True` | Django debug mode |
| `DJANGO_SECRET_KEY` | `dev-secret-key...` | Django secret key |

---

## Troubleshooting

**Models not trained:**
Run `make train` before starting the backend.

**ARIMA convergence warnings:**
These are expected during grid search — rejected orders are silently discarded.

**Torch import errors:**
Ensure PyTorch CPU version is installed: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**Database errors after model changes:**
Delete `backend/db.sqlite3` and re-run `make migrate`.
