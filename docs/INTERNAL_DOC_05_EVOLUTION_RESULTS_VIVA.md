# MASTER INTERNAL DOCUMENTATION — Part 5
# Sections 10–14: Evolution, Results, Updates, Future Work, Viva Prep

---

# SECTION 10 — RESEARCH & IMPLEMENTATION EVOLUTION

## 10.1 Key Transitions

### CPU-Only → Multi-Resource
- **Phase 1**: All models treated workload as single float (CPU%)
- **Phase 2**: All models accept (n, 3) arrays: [CPU, Memory, Network I/O]
- **Impact**: LSTM architecture changed from Input(20,1)→Output(1) to Input(20,3)→Output(5,3)
- **Rationale**: Real cloud scheduling considers multiple resource dimensions simultaneously

### Basic Forecasting → Multi-Model Pipeline
- **Phase 1**: LSTM, ARIMA, and Combined as separate scripts
- **Phase 2**: Unified `train_all.py` pipeline with 5 sequential components + `training_summary.json`
- **Added**: GBR as explicit multi-output baseline; AnomalyDetector as 5th component

### Simulation Scripts → Full-Stack Application
- **Phase 1**: `python evaluate.py` → CSV + PNG output
- **Phase 2**: Django REST API + React dashboard with persistent database
- **Impact**: Results are now queryable, comparable, and visually explorable

### Standalone Models → Integrated ML Pipeline
- **Phase 1**: Each model loaded independently
- **Phase 2**: `inference.py` provides unified `predict()` with singleton caching and `invalidate_cache()` after retraining

### Simple Scheduler → Anomaly-Aware Scheduler
- **Phase 1**: CPU>70% → scale up (reactive); predicted>55% → scale up (predictive)
- **Phase 2**: Multi-resource thresholds with consecutive-step confirmation (reactive); forecast-aware + anomaly-threshold-reduction + spike-acceleration-fast-path (predictive)

## 10.2 Important Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| Synthetic data over real telemetry | Reproducibility, controlled experiments, no external dependencies |
| CPU-only PyTorch (no GPU) | All models train in <5 min on CPU; no CUDA dependency for evaluators |
| SQLite over PostgreSQL | Zero-config for demo/viva; trivial migration path exists |
| Monochrome theme | Academic aesthetic matching IEEE paper grayscale |
| Per-resource ARIMA (not VAR) | Each resource has distinct temporal dynamics; independent models are simpler and more robust |
| OR logic for anomaly detection | High recall priority — better to over-flag than miss anomalies |
| CombinedForecaster for scheduler | Tests the ensemble hypothesis directly in the scheduling loop |
| MinMaxScaler (not StandardScaler) | Workload values are naturally bounded [0, 100] |

## 10.3 Lessons Learned

1. **Ensemble ≠ always better**: Combined model (R²=0.47) underperformed individual LSTM (R²=0.93) — phase-shift interference
2. **Per-resource weighting matters**: ARIMA dominates CPU (lower RMSE) while LSTM dominates Network
3. **Anomaly detection adds safety**: 10% threshold reduction during anomalies provides meaningful buffer
4. **Rate-of-change detection supplements forecasting**: Spike fast-path catches sudden acceleration before forecast window updates
5. **Backward compatibility is critical**: All models accept 1-D input for legacy compatibility

---

# SECTION 11 — RESULTS & EVALUATION EVOLUTION

## 11.1 Phase 1 Results (CPU-Only, for Reference)

### Model Performance
| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| LSTM | 0.9696 | 5.11 | 4.04 |
| ARIMA | 0.6351 | 1.85 | 0.93 |
| Combined | 0.7952 | 14.53 | 6.56 |

### Scheduler Comparison
| Pattern | Reactive Overloads | Predictive Overloads | Reduction |
|---------|-------------------|---------------------|-----------|
| Gradual | 8 | 3 | 62% |
| Spike | 31 | 19 | 39% |
| Periodic | 39 | 20 | 49% |

## 11.2 Phase 2 Results (Multi-Resource)

### Model Performance (3-Signal Forecasting)
| Model | Overall R² | Overall RMSE | CPU R² | Memory R² | Network R² |
|-------|-----------|-------------|--------|-----------|------------|
| **LSTM** | **0.9287** | 7.893 | 0.9632 | 0.9520 | 0.8657 |
| GBR | 0.9158 | 7.998 | 0.9638 | 0.9427 | 0.8341 |
| ARIMA | 0.5624 | 7.127 | 0.8741 | 0.5911 | 0.2185 |
| Combined | 0.4698 | 5.789 | 0.6188 | 0.3134 | 0.0907 |

### Anomaly Detection
- 30 anomalies detected out of 600 training steps (5.0%)
- IsolationForest + Rolling Z-Score dual strategy
- Predictive scheduler dynamically lowers thresholds by 10pp during anomaly

## 11.3 Why Certain Models Performed As They Did

### LSTM: Best Overall (R²=0.929)
- Multi-layer LSTM captures non-linear temporal dependencies across all 3 resources
- BatchNorm stabilises training despite varied workload magnitudes
- CosineAnnealing LR helps escape local minima

### ARIMA: Good CPU, Poor Network (R²=0.562)
- CPU signal has strong linear trend → ARIMA excels
- Network I/O has stochastic bursts (Poisson spikes) → ARIMA cannot model
- Memory has lagged correlation → moderate ARIMA performance

### Combined: Underperformed (R²=0.470)
- Evaluated on separate test segment where phase misalignment accumulates
- Weighting two noisy predictions doesn't reduce variance
- **Valid research finding**: simple ensemble doesn't always improve over best individual model

### GBR: Strong Baseline (R²=0.916)
- Gradient boosting handles non-linear relationships well
- 200 trees with depth 5 capture complex patterns without overfitting
- Fast inference makes it ideal for real-time scheduling

## 11.4 Evaluation Methodology Evolution

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Resources evaluated | CPU only | CPU + Memory + Network I/O |
| Metrics | R², RMSE, MAE | Per-resource R²/RMSE/MAE + overall |
| Scheduler metrics | Overload count, reduction % | + per-resource overload breakdown, cost, scaling counts |
| Anomaly evaluation | None | Anomaly rate, z-score/iforest breakdown |
| Persistence | CSV files | SQLite database |
| Visualisation | Matplotlib PNGs | Interactive Chart.js dashboard |
| Formal evaluation | CLI script | POST /api/evaluation/run/ with DB persistence |

---

# SECTION 12 — WHAT MUST BE UPDATED IN REPORT & PPT

## 12.1 Report Update Checklist

| Section | Status | Action Required |
|---------|--------|-----------------|
| **Abstract** | ⚠️ Outdated | Update to mention multi-resource scheduling, anomaly detection, full-stack architecture |
| **Introduction** | ⚠️ Outdated | Add Phase 2 scope, mention 3 resource dimensions |
| **Literature Review** | ✅ Likely OK | May need anomaly detection references |
| **System Architecture** | ❌ Outdated | Replace with 3-tier architecture (Frontend→Backend→ML) |
| **Architecture Diagrams** | ❌ Outdated | Redraw with React + Django + SQLite + Docker |
| **ML Methodology** | ❌ Outdated | Rewrite for multi-resource models, add GBR, add anomaly detector |
| **LSTM Architecture** | ❌ Outdated | Update: Input(20,3)→LSTM→BN→FC→FC→Reshape(5,3) |
| **ARIMA Description** | ⚠️ Outdated | Update: 3× independent per-channel with AIC grid search |
| **Combined Ensemble** | ❌ Outdated | Rewrite: per-resource inverse-RMSE weights (not scalar) |
| **Scheduler Logic** | ❌ Outdated | Rewrite: multi-resource thresholds, anomaly-aware adjustment |
| **Workload Generation** | ❌ Outdated | Update: 3-signal generation with correlation formulas |
| **Database Design** | ❌ Missing | Add: 7-table SQLite schema |
| **API Architecture** | ❌ Missing | Add: 21 REST endpoints |
| **Frontend Design** | ❌ Missing | Add: 8-page React dashboard |
| **Docker Deployment** | ❌ Missing | Add: 3-service docker-compose |
| **Results Tables** | ❌ Outdated | Replace with Phase 2 per-resource metrics |
| **Overload Results** | ⚠️ Partial | Keep Phase 1 results as reference, add Phase 2 multi-resource |
| **Screenshots** | ❌ Outdated | Capture all 8 dashboard pages |
| **Anomaly Detection** | ❌ Missing | Add: IsolationForest + Z-Score methodology |
| **Conclusion** | ⚠️ Outdated | Update with Phase 2 contributions |
| **Future Work** | ⚠️ Outdated | Update with current pending items |

## 12.2 PPT Update Checklist

| Slide Topic | Action |
|-------------|--------|
| Title slide | Update year if needed |
| Problem statement | Add multi-resource angle |
| Architecture diagram | **Replace**: new 3-tier diagram with Docker |
| ML model slides | **Rewrite**: multi-resource architectures |
| LSTM slide | Update architecture diagram with (20,3)→(5,3) |
| ARIMA slide | Update to per-channel with AIC grid search |
| Combined slide | Add per-resource weighting formula |
| GBR slide | **Add**: MultiOutput baseline explanation |
| Anomaly detection | **Add new slide**: IsolationForest + Z-Score |
| Scheduler comparison | Update with multi-resource thresholds |
| Results tables | **Replace** with Phase 2 per-resource metrics |
| Dashboard screenshots | **Replace all**: capture 8 pages from current UI |
| Docker/deployment | **Add new slide**: 3-service architecture |
| Demo flow | Update: show live dashboard workflow |
| Future work | Update with current pending items |

## 12.3 Architecture Diagrams Needed

| Diagram | Description |
|---------|-------------|
| System architecture | 3-tier: Frontend ↔ Backend ↔ ML Pipeline + DB |
| Docker architecture | 3-service: trainer → backend → frontend |
| ML training pipeline | train_all.py → 5 sequential components |
| LSTM architecture | Input(20,3)→LSTM(2L)→BN→FC→FC→Reshape(5,3) |
| Data flow | Workload generation → training → inference → scheduling |
| Scheduler decision flow | Reactive vs Predictive side-by-side flowchart |
| Database ER diagram | 7 tables with foreign key relationships |
| Anomaly detection | Dual strategy: Z-Score OR IsolationForest |

## 12.4 Screenshots Needed

| Page | Priority | Content to Capture |
|------|----------|-------------------|
| DashboardPage | High | Model status cards, KPI overview |
| SimulationPage | Medium | Generated workload chart |
| TrainingPage | Medium | Training in progress / completed |
| FindingsPage | **Critical** | Scheduler comparison chart + overload breakdown |
| ModelComparisonPage | High | Per-resource R² comparison + forecast chart |
| MetricsPage | Medium | Aggregated stats |
| AnomalyLogPage | Medium | Anomaly log table + summary stats |
| Docker terminal | Medium | `docker compose up` output |

---

# SECTION 13 — FUTURE WORK & PENDING TASKS

## 13.1 Future Research Directions

| Area | Description | Difficulty |
|------|-------------|------------|
| **Kubernetes deployment** | Replace simulated scaling with real K8s HPA integration | High |
| **Reinforcement learning** | DQN/PPO agent for scheduling decisions instead of threshold-based | High |
| **Online learning** | Incremental model updates from streaming telemetry | Medium |
| **Real telemetry** | Ingest actual Prometheus/CloudWatch metrics | Medium |
| **Transformer models** | Replace LSTM with attention-based architecture | Medium |
| **SLA-aware scheduling** | Incorporate SLA violation penalties into cost function | Medium |
| **Energy-aware scheduling** | Add power consumption as 4th resource dimension | Medium |
| **Distributed inference** | Serve models via TorchServe/TFServing for production | High |
| **Multi-tenant scheduling** | Per-tenant resource allocation and QoS | High |

## 13.2 Technical Debt

| Item | Priority | Description |
|------|----------|-------------|
| Root `workload_generator.py` | Low | Deprecated duplicate — should be removed |
| `Dockerfile.legacy` | Low | Old single-container — should be removed |
| ComparisonPage vs FindingsPage | Low | Some overlap in scheduler comparison functionality |
| Test suite | Medium | No unit tests or integration tests exist |
| Error handling | Low | Some API endpoints lack detailed error responses |
| WebSocket for training | Low | Training is synchronous; could use async + progress streaming |

## 13.3 Optimisation Opportunities

| Area | Opportunity |
|------|-------------|
| ARIMA training | Parallel per-channel fitting (currently sequential) |
| GBR model size | 13 MB is large; consider model compression |
| Frontend bundle | Tree-shake unused Chart.js components |
| Database | Add indexes on SchedulerRun(scheduler_type, pattern) |
| Docker image | Use multi-stage Python build to reduce image size |

---

# SECTION 14 — VIVA & DEMO PREPARATION NOTES

## 14.1 Demo Flow (Recommended)

1. **Start**: Show Docker deployment (`docker compose up --build`)
2. **Dashboard**: Point out model status cards, project overview
3. **Simulation**: Generate a "combined" workload (200 steps, seed 42)
4. **Training**: Show "Train All" button, explain 5-component pipeline
5. **Findings** (★ KEY SLIDE): Run scheduler comparison → show overload reduction %
6. **Model Comparison**: Show per-resource R² for all 4 models
7. **Anomaly Log**: Show anomaly detection results
8. **Architecture**: Walk through the 3-tier system design

## 14.2 Strong Technical Talking Points

1. **Multi-resource forecasting**: "We forecast CPU, Memory, and Network I/O simultaneously using a single LSTM architecture"
2. **Per-resource ensemble weights**: "Our combined model uses resource-specific weights — ARIMA may dominate CPU while LSTM dominates Network"
3. **Anomaly-aware scheduling**: "When anomalies are detected, we proactively lower thresholds by 10% as a safety margin"
4. **Spike fast-path**: "We detect rate-of-change acceleration to respond to sudden spikes before the forecast window updates"
5. **Honest results**: "Our ensemble underperformed individual LSTM — this is a valid finding that ensemble ≠ always better"
6. **Production-ready architecture**: "Full REST API, database persistence, Docker deployment"

## 14.3 Likely Viva Questions & Answers

### Q: Why did you choose LSTM over Transformer/Attention?
**A**: LSTM is well-established for time-series forecasting with relatively small datasets. Our training data (~4K steps) would undertrain a Transformer. LSTM's recurrent state captures temporal dependencies efficiently at this scale. Transformer-based models are identified as future work.

### Q: Why synthetic data instead of real cloud telemetry?
**A**: Synthetic data provides: (1) full reproducibility with seed control, (2) controlled experiments across 4 distinct patterns, (3) no external API dependencies. Real telemetry ingestion is planned as future work.

### Q: Why does the Combined ensemble perform worse than LSTM alone?
**A**: Phase misalignment between LSTM and ARIMA predictions causes destructive interference when weighted-averaged. ARIMA's one-step walk-forward methodology introduces different temporal error patterns than LSTM's sliding-window approach. This is a valid research finding documented in ensemble literature.

### Q: Why SQLite and not a production database?
**A**: SQLite provides zero-configuration persistence ideal for academic demonstration. The Django ORM ensures standard SQL — migrating to PostgreSQL requires changing one line in `settings.py`. Our data volumes (~10K rows per experiment) are well within SQLite's capabilities.

### Q: How does anomaly detection integrate with scheduling?
**A**: The AnomalyDetector uses IsolationForest + Rolling Z-Score (OR logic). When an anomaly is detected during a predictive scheduler step, all scale-up thresholds are temporarily reduced by 10 percentage points (e.g., 65% → 58.5%), providing a proactive safety margin during unusual workload events.

### Q: What is the significance of the GBR model?
**A**: GBR serves as a fast, non-deep-learning baseline for comparison. It validates that our LSTM provides genuine improvement over a strong tree-based model. GBR achieves R²=0.916 vs LSTM's R²=0.929 — competitive but LSTM captures finer temporal patterns.

### Q: How are memory and network signals generated?
**A**: Memory follows CPU with a 3-step temporal lag (ρ≈0.75): `memory(t) = 0.7×cpu(t-3) + 0.3×cpu(t) + N(0,5)`. Network is proportional with Poisson burst noise: `network(t) = cpu(t)×0.9 + Poisson(λ=0.05)×Uniform(15,40) + N(0,8)`. These correlations reflect empirical cloud resource relationships.

### Q: What makes your predictive scheduler "proactive"?
**A**: Three mechanisms: (1) Lower thresholds (65% vs 80%) justified because predictions allow early action; (2) 5-step-ahead forecasting detects overload before it occurs; (3) Rate-of-change acceleration detection for sudden spikes. Combined, these ensure scaling happens before — not after — overload.

### Q: How would this work in production?
**A**: Replace synthetic workload with real Prometheus/CloudWatch metrics ingestion. Deploy models via TorchServe. Replace simulated scaling with Kubernetes HPA API calls. The REST API architecture is already production-ready.

### Q: What is the overload reduction percentage?
**A**: Phase 1 (CPU-only): 39-62% reduction across patterns. Phase 2 extends this to multi-resource scheduling where the predictive scheduler proactively prevents overload across all three resource dimensions simultaneously.

### Q: Why 4 models instead of just the best one?
**A**: Comparative evaluation is a core research contribution. We demonstrate that (1) LSTM outperforms ARIMA for non-linear patterns, (2) GBR provides a strong non-DL baseline, (3) ensemble doesn't always improve on individual models, and (4) different models excel on different resource signals. This multi-model comparison strengthens the research validity.

### Q: What is CAPACITY_PER_UNIT and why 10?
**A**: Each "resource unit" provides 10 workload units of capacity. At 100% utilisation, a single unit serves workload=10. This abstraction mirrors real cloud instance sizing. The value 10 was chosen for interpretable CPU utilisation percentages: workload=5 on 1 unit = 50% CPU.
