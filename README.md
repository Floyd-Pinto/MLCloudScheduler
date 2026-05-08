# ML-Based Adaptive Cloud Resource Scheduling

> **Major Project — 2025-2026** | B.Tech CSE | Sardar Patel Institute of Technology, Mumbai
> Team: Floyd Pinto · Adveat Sankhe · Powromita Ughade | Guide: Prof. Anand Godbole

---

## Research Question

> Can ML-based predictive multi-resource scheduling (LSTM, ARIMA, Combined ensemble)
> reduce cloud resource overload events compared to threshold-based reactive autoscaling
> across CPU, Memory, and Network I/O simultaneously?

**Phase 1 Result**: 39–62% overload reduction on CPU-only scheduling.
**Phase 2 Extension**: Multi-resource (CPU + Memory + Network I/O) predictive scheduling
with anomaly detection.

---

## Phase 2 Key Results

### Model Performance (Multi-Resource, 3-Signal Forecasting)

| Model    | Overall R² | Overall RMSE | CPU R² | Memory R² | Network R² |
|----------|------------|--------------|--------|-----------|------------|
| **LSTM** | **0.9287** | 7.893        | ~0.96  | ~0.94     | ~0.89      |
| **ARIMA**| 0.5624     | 7.127        | ~0.61  | ~0.55     | ~0.52      |
| **GBR**  | 0.9158     | 7.998        | ~0.96  | ~0.94     | ~0.83      |
| **Combined** | 0.4698 | 5.789       | —      | —         | —          |

### Anomaly Detection
- 30 anomalies detected (5.0% of steps) using IsolationForest + Rolling Z-Score
- Predictive scheduler dynamically lowers thresholds by 10pp when anomaly detected

### Phase 1 Results (CPU-Only, for Reference)

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
|-------|------------|---------|
| Frontend | React 19, Vite 8, Chart.js | Research dashboard |
| Backend | Django 5, Django REST Framework | REST API + data persistence |
| ML | PyTorch (LSTM), Statsmodels (ARIMA), Scikit-learn (GBR, IsolationForest) | Forecasting + anomaly detection |
| Database | SQLite | Zero-config persistence |
| Infra | Docker, Docker Compose | Containerised 3-service setup |

---

## Quick Start

### Docker (Recommended)
```bash
git clone https://github.com/Floyd-Pinto/MLCloudScheduler.git
cd MLCloudScheduler
docker compose up --build
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000/api/
```
Models are trained automatically before the backend starts (~3-5 min first run).

### Manual
```bash
# See RUNNING.md for full step-by-step guide
make install   # pip + npm install
make train     # train all 5 model components (~3-4 min)
make migrate   # apply DB migrations
make dev       # start backend (:8000) + frontend (:5173)
```

---

## Project Structure

```
MLCloudScheduler/
├── model/                      ← ML pipeline
│   ├── workload_generator.py   ← 3-resource synthetic workload (CPU/Memory/Network)
│   ├── lstm_model.py           ← Multi-input/output LSTM (20×3 → 5×3)
│   ├── arima_model.py          ← Per-channel ARIMA (3 independent models)
│   ├── combined_model.py       ← Per-resource inverse-RMSE ensemble
│   ├── train_gbr.py            ← MultiOutput GBR (60-feature input, 15-value output)
│   ├── train_all.py            ← Master training pipeline (5 components)
│   ├── anomaly_detector.py     ← IsolationForest + Rolling Z-Score
│   ├── inference.py            ← Unified prediction interface
│   ├── reactive_scheduler.py   ← 3-resource threshold-based scheduler
│   ├── predictive_scheduler.py ← ML-forecast + anomaly-aware scheduler
│   └── saved_models/           ← 14 trained model artifacts
├── backend/                    ← Django REST API
│   ├── simulation/             ← Workload generation
│   ├── ml_model/               ← Training & inference
│   ├── scheduler/              ← Scheduler comparison
│   ├── metrics/                ← Aggregated stats
│   ├── evaluation/             ← Evaluation persistence
│   └── anomaly/                ← Anomaly detection logs (NEW)
├── frontend/                   ← React dashboard (8 pages)
│   └── src/pages/
│       ├── DashboardPage.jsx
│       ├── SimulationPage.jsx  ← Live Mode toggle
│       ├── TrainingPage.jsx
│       ├── FindingsPage.jsx    ← Per-resource overload breakdown + PNG export
│       ├── MetricsPage.jsx
│       ├── LogsPage.jsx
│       ├── AnomalyLogPage.jsx  ← NEW
│       └── ModelComparisonPage.jsx ← NEW — per-resource metrics
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml          ← 3-service: trainer → backend → frontend
├── RUNNING.md                  ← Full setup guide
└── docs/
    ├── MODELS.md               ← All 5 model architectures
    ├── API.md
    ├── ARCHITECTURE.md
    └── SETUP.md
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/simulation/generate/` | Generate 3-resource workload |
| GET  | `/api/simulation/runs/` | List workload runs |
| POST | `/api/ml/train/` | Train model (`model_type`: lstm/arima/combined/gbr/all) |
| GET  | `/api/ml/status/` | Model readiness + per-resource metrics |
| POST | `/api/ml/compare-models/` | Compare all models on a workload |
| POST | `/api/scheduler/compare/` | Run reactive vs predictive comparison |
| GET  | `/api/metrics/summary/` | Aggregated stats |
| GET  | `/api/anomaly/logs/` | Anomaly log entries |
| GET  | `/api/anomaly/summary/` | Anomaly detection summary stats |

Full docs: [docs/API.md](docs/API.md)

---

## Documentation

| Document | Path | Content |
|---|---|---|
| **README** | `README.md` | Project overview, structure, quick start |
| **Running Guide** | `RUNNING.md` | Full setup, Docker, and manual deployment guide |
| **Setup Guide** | `docs/SETUP.md` | Step-by-step installation |
| **Architecture** | `docs/ARCHITECTURE.md` | System design, ML pipeline, design decisions |
| **API Reference** | `docs/API.md` | All REST endpoints with request/response examples |
| **Model Details** | `docs/MODELS.md` | ML model architectures, hyperparameters, training pipeline |
