# MASTER INTERNAL DOCUMENTATION — Part 1
# Sections 1–2: Evolution Summary & Implementation Status

> **ML-Based Adaptive Cloud Resource Scheduling**
> Floyd Pinto · Adveat Sankhe · Powromita Ughade | Guide: Prof. Anand Godbole
> Sardar Patel Institute of Technology, Mumbai | Generated: 2026-05-11

---

# SECTION 1 — PROJECT EVOLUTION SUMMARY

## 1.1 Original Project Scope (Phase 1)

- **Title**: ML-Based Cloud Resource Scheduling
- **Focus**: CPU-only workload forecasting using LSTM + ARIMA
- **Scheduler**: Single-resource reactive vs predictive comparison
- **ML Models**: LSTM (single-signal), ARIMA (single-signal), Combined (weighted average)
- **Output**: CLI-based simulation scripts producing CSV/PNG outputs
- **No frontend, no backend, no database, no Docker**
- **Workload**: Single CPU utilisation signal (0–100%)
- **Evaluation**: Overload count comparison across 3 patterns (gradual, spike, periodic)

### Original Architecture
```
workload_generator.py → reactive_scheduler.py / predictive_scheduler.py
                      → metrics_collector.py → CSV + matplotlib PNGs
```

### Original ML Approach
- LSTM: Single-input single-output, window=20 → predict 5 steps
- ARIMA: Single-signal ARIMA(2,1,2) with walk-forward validation
- Combined: Scalar inverse-RMSE weighted average (w_lstm, w_arima)
- GBR: Basic GradientBoosting as fast scheduler inference backbone

### Original Limitations
- CPU-only: ignored memory and network I/O
- No anomaly detection
- No web dashboard — results only via CLI/CSV/PNG
- No persistent storage — results lost between runs
- No containerisation
- No REST API
- Combined ensemble used global scalar weights, not per-resource

## 1.2 How the Project Evolved

### Phase 1 → Phase 2 Transition Drivers

| Driver | Rationale |
|--------|-----------|
| Multi-resource gap | Real clouds monitor CPU + Memory + Network; CPU-only was unrealistic |
| Anomaly blindness | Phase 1 had no mechanism for detecting/handling abnormal workload spikes |
| No persistence | Results vanished on restart; couldn't track experiments over time |
| Demo limitations | CLI output inadequate for viva/proposal defense presentation |
| Reproducibility | No API meant no standardised way to trigger experiments |
| Deployment gap | No containerisation meant complex manual setup for evaluators |

### Key Changes Made

1. **Multi-Resource Forecasting** — All models upgraded from 1-D to 3-D (CPU, Memory, Network I/O)
2. **Anomaly Detection System** — IsolationForest + Rolling Z-Score (dual strategy, OR logic)
3. **Full-Stack Architecture** — Django REST backend + React/Vite frontend dashboard
4. **Database Persistence** — SQLite with 7 ORM models for all experiment data
5. **Docker Deployment** — 3-service docker-compose (trainer → backend → frontend)
6. **Per-Resource Ensemble** — Combined model now uses per-resource inverse-RMSE weights
7. **Anomaly-Aware Scheduler** — Predictive scheduler lowers thresholds by 10% during anomalies
8. **Research Dashboard** — 8-page monochrome academic UI with Chart.js visualisations

## 1.3 Comparison Tables

### Original vs Current

| Aspect | Phase 1 (Original) | Phase 2 (Current) |
|--------|--------------------|--------------------|
| Resource signals | CPU only (1-D) | CPU + Memory + Network I/O (3-D) |
| LSTM architecture | Input(20,1)→LSTM→FC→1 | Input(20,3)→LSTM(2L,128)→BN→FC→FC→Reshape(5,3) |
| ARIMA | Single ARIMA(2,1,2) | 3× independent ARIMA per channel, AIC grid search |
| GBR | window=10→predict 1 | window=20×3=60 features→predict 5×3=15 values |
| Combined | Scalar weights | Per-resource inverse-RMSE weights |
| Anomaly detection | None | IsolationForest + Rolling Z-Score |
| Scheduler (reactive) | CPU>70% → scale up | CPU>80% OR Mem>80% OR Net>85% for 2 consec steps |
| Scheduler (predictive) | predicted_cpu>55% | Multi-resource forecast + anomaly-aware threshold |
| Frontend | None | React 19 + Vite 8, 8 pages, Chart.js |
| Backend | None | Django 6 + DRF, 6 apps, 15+ API endpoints |
| Database | None | SQLite, 7 tables, Django ORM |
| Docker | None | 3-service docker-compose |
| Deployment | Manual Python scripts | `make dev` or `docker compose up --build` |
| Output format | CSV + PNG files | REST API JSON + interactive dashboard |

### Planned vs Implemented Features

| Feature | Planned | Implemented | Notes |
|---------|---------|-------------|-------|
| CPU forecasting | ✅ | ✅ | Extended to multi-resource |
| LSTM model | ✅ | ✅ | Upgraded to multi-input/output |
| ARIMA model | ✅ | ✅ | Per-channel with AIC selection |
| Combined ensemble | ✅ | ✅ | Per-resource weights (novel) |
| GBR baseline | ✅ | ✅ | MultiOutput, 60→15 |
| Reactive scheduler | ✅ | ✅ | Multi-resource thresholds |
| Predictive scheduler | ✅ | ✅ | Anomaly-aware + spike fast-path |
| Overload comparison | ✅ | ✅ | Extended with per-resource breakdown |
| Memory forecasting | ❌ | ✅ | Added in Phase 2 |
| Network I/O forecasting | ❌ | ✅ | Added in Phase 2 |
| Anomaly detection | ❌ | ✅ | IsolationForest + Z-Score |
| REST API | ❌ | ✅ | 15+ endpoints |
| Web dashboard | ❌ | ✅ | 8-page React app |
| Database persistence | ❌ | ✅ | SQLite with 7 tables |
| Docker deployment | ❌ | ✅ | 3-service compose |
| Kubernetes | ❌ | ❌ | Future work |
| Reinforcement learning | ❌ | ❌ | Future work |
| Real telemetry | ❌ | ❌ | Uses synthetic data |

### Removed/Replaced Components

| Component | Status | Replacement |
|-----------|--------|-------------|
| Standalone `workload_generator.py` (root) | Kept for backward compat | `model/workload_generator.py` is authoritative |
| Single-signal generators | Kept as internal functions | `generate_multivariate()` wraps them |
| Scalar ensemble weights | Replaced | Per-resource inverse-RMSE dict weights |
| CLI-only evaluation | Replaced | REST API + Dashboard |
| `Dockerfile.legacy` | Deprecated | `Dockerfile.backend` + `Dockerfile.frontend` |
| Matplotlib output PNGs | Kept in `outputs/` | Dashboard Chart.js is primary visualisation |

---

# SECTION 2 — CURRENT IMPLEMENTATION STATUS

## 2.1 Module Status Overview

| Module | Status | Working | Notes |
|--------|--------|---------|-------|
| **model/workload_generator.py** | ✅ Done | 4 patterns × 3 resources | Fully functional |
| **model/lstm_model.py** | ✅ Done | Train + inference + save/load | Multi-resource, R²=0.929 |
| **model/arima_model.py** | ✅ Done | Per-channel AIC grid search | R²=0.562 overall |
| **model/combined_model.py** | ✅ Done | Per-resource weighted ensemble | R²=0.470 overall |
| **model/train_gbr.py** | ✅ Done | MultiOutput GBR | R²=0.916 overall |
| **model/anomaly_detector.py** | ✅ Done | IsolationForest + Z-Score | 5% contamination |
| **model/inference.py** | ✅ Done | Unified predict() for all 4 models | Singleton caching |
| **model/reactive_scheduler.py** | ✅ Done | Multi-resource thresholds | Consec-step logic |
| **model/predictive_scheduler.py** | ✅ Done | Forecast + anomaly-aware | Spike fast-path |
| **model/metrics_collector.py** | ✅ Done | Per-step multi-resource records | Dataclass-based |
| **model/evaluate.py** | ✅ Done | CLI comparison runner | Used by backend |
| **model/train_all.py** | ✅ Done | 5-component sequential pipeline | ~3-5 min |
| **backend/simulation/** | ✅ Done | Generate + list + get + delete | Multi-resource |
| **backend/scheduler/** | ✅ Done | Reactive + predictive + compare | Anomaly logging |
| **backend/ml_model/** | ✅ Done | Train + status + predict + compare | All 4 models |
| **backend/metrics/** | ✅ Done | List + summary aggregation | Filter by type/pattern |
| **backend/evaluation/** | ✅ Done | Run + list + comparison | Formal persistence |
| **backend/anomaly/** | ✅ Done | Logs + summary | Populated during scheduler runs |
| **frontend/DashboardPage** | ✅ Done | Overview with model status cards | KPI display |
| **frontend/SimulationPage** | ✅ Done | Generate workloads, view chart | Live mode toggle |
| **frontend/TrainingPage** | ✅ Done | Train individual/all models | Status tracking |
| **frontend/FindingsPage** | ✅ Done | Scheduler comparison + overload breakdown | PNG export |
| **frontend/ModelComparisonPage** | ✅ Done | Per-resource model metrics + charts | API-linked |
| **frontend/MetricsPage** | ✅ Done | Aggregated reactive vs predictive stats | — |
| **frontend/LogsPage** | ✅ Done | Step-by-step scheduler action viewer | — |
| **frontend/AnomalyLogPage** | ✅ Done | Anomaly detection log viewer | — |
| **Docker setup** | ✅ Done | 3-service compose | trainer→backend→frontend |
| **Makefile** | ✅ Done | 10 targets | install/train/dev/docker |

## 2.2 API Endpoint Status

| Endpoint | Method | Status | Frontend Integration |
|----------|--------|--------|---------------------|
| `/api/simulation/generate/` | POST | ✅ | SimulationPage |
| `/api/simulation/runs/` | GET | ✅ | SimulationPage |
| `/api/simulation/runs/{id}/` | GET/DELETE | ✅ | SimulationPage |
| `/api/ml/train/` | POST | ✅ | TrainingPage |
| `/api/ml/status/` | GET | ✅ | DashboardPage, TrainingPage |
| `/api/ml/predict/` | POST | ✅ | Internal |
| `/api/ml/predict-all/` | POST | ✅ | Internal |
| `/api/ml/compare-models/` | POST/GET | ✅ | ModelComparisonPage |
| `/api/ml/history/` | GET | ✅ | TrainingPage |
| `/api/scheduler/reactive/` | POST | ✅ | Internal |
| `/api/scheduler/predictive/` | POST | ✅ | Internal |
| `/api/scheduler/compare/` | POST | ✅ | FindingsPage |
| `/api/scheduler/runs/` | GET | ✅ | LogsPage |
| `/api/scheduler/runs/{id}/` | GET | ✅ | LogsPage |
| `/api/metrics/` | GET | ✅ | MetricsPage |
| `/api/metrics/summary/` | GET | ✅ | MetricsPage, DashboardPage |
| `/api/evaluation/run/` | POST | ✅ | FindingsPage |
| `/api/evaluation/` | GET | ✅ | FindingsPage |
| `/api/evaluation/comparison/` | GET | ✅ | FindingsPage |
| `/api/anomaly/logs/` | GET | ✅ | AnomalyLogPage |
| `/api/anomaly/summary/` | GET | ✅ | AnomalyLogPage |

## 2.3 System Readiness

| System | Ready for Demo | Ready for Viva | Notes |
|--------|---------------|----------------|-------|
| ML Training Pipeline | ✅ | ✅ | All 5 components train successfully |
| Multi-Resource Forecasting | ✅ | ✅ | 3-signal simultaneous prediction |
| Scheduler Comparison | ✅ | ✅ | Side-by-side with overload reduction |
| Anomaly Detection | ✅ | ✅ | Logs populated, summary available |
| REST API | ✅ | ✅ | All 21 endpoints functional |
| React Dashboard | ✅ | ✅ | 8 pages, all connected to API |
| Docker Deployment | ✅ | ✅ | Single `docker compose up` |
| Documentation | ✅ | ✅ | README, RUNNING, docs/ folder |
