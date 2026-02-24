# ML-Based Adaptive Cloud Resource Scheduling

A simulated cloud resource scheduling system that uses machine learning to **proactively** scale resources before overload occurs, compared against a traditional reactive (threshold-based) scheduler.

## Project Structure

```
ml-cloud-scheduler/
├── main.py                       # Entry point — runs the full simulation & comparison
├── workload_generator.py         # Synthetic workload patterns (gradual, spike, periodic, combined)
├── schedulers/
│   ├── reactive_scheduler.py     # Baseline: threshold-based reactive scaling
│   └── predictive_scheduler.py   # Proposed: ML-based predictive scaling (GradientBoosting)
├── metrics/
│   └── collector.py              # Records CPU usage, overload events, cost estimates
├── visualization/
│   └── plotter.py                # Comparison charts (capacity, CPU%, summary bar)
├── data/                         # Generated simulation CSVs (git-ignored)
├── models/                       # Saved ML model artifacts — gbr_model.pkl, scaler.pkl
├── outputs/                      # Generated PNG plots (git-ignored)
├── Dockerfile                    # Multi-stage Docker build (python:3.12-slim)
├── docker-compose.yml            # Compose file — mounts data/, models/, outputs/ to host
├── Makefile                      # Convenience shortcuts (venv, install, run, docker-*)
├── requirements.txt
├── .gitignore
└── .dockerignore
```

## Scheduling Approaches

| Approach | Mechanism | Scaling Trigger | Scale-up threshold |
|---|---|---|---|
| **Reactive (Baseline)** | Static CPU thresholds | After overload detected | 70% CPU |
| **Predictive (Proposed)** | GradientBoosting time-series forecast | Before overload occurs | 60% predicted CPU |

The predictive scheduler uses a sliding window of the last 10 load observations to forecast load 5 steps ahead, retraining every 10 steps. During the initial warmup period (before enough history is collected) it falls back to reactive logic so there is no cold-start gap.

## Simulation Results

Predictive scheduling reduces overload events across all workload patterns:

| Pattern | Reactive overloads | Predictive overloads | Improvement |
|---|---|---|---|
| Gradual | 8 | **4** | −50% |
| Spike | 31 | **24** | −23% |
| Periodic | 39 | **20** | −49% |
| Combined | 21 | **19** | −10% |

Each capacity unit handles 10 workload units (CPU % = `workload / (capacity × 10) × 100`). Overload is declared when CPU > 80%.

## Getting Started

### Option A — Local virtual environment

```bash
# 1. Create and activate venv
python3 -m venv .venv
source .venv/bin/activate        # bash/zsh
# source .venv/bin/activate.fish # fish shell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run simulation
python main.py
```

Or use the Makefile:

```bash
make venv     # create .venv
make install  # install deps
make run      # run simulation
```

### Option B — Docker

```bash
# Build image
make docker-build   # or: docker compose build

# Run simulation (outputs written to ./data, ./models, ./outputs on the host)
make docker-run     # or: docker compose run --rm scheduler
```

Outputs and plots are saved to `data/`, `models/`, and `outputs/` after each run.

## Makefile Targets

| Target | Description |
|---|---|
| `make venv` | Create `.venv` virtual environment |
| `make install` | Install Python dependencies into `.venv` |
| `make run` | Run full simulation locally |
| `make docker-build` | Build the Docker image |
| `make docker-run` | Run simulation inside Docker |
| `make clean` | Remove generated CSVs, model files, and plots |

## Technologies Used

- **Machine Learning** — `GradientBoostingRegressor` (scikit-learn) for time-series load forecasting
- **Simulation** — Synthetic workload generation (gradual, spike, periodic, combined patterns)
- **Visualization** — Matplotlib / Seaborn comparison plots (capacity, CPU%, summary bar charts)
- **Infrastructure** — Docker multi-stage build, Docker Compose with volume mounts
- **Concepts** — OS CPU scheduling, cloud autoscaling, adaptive feedback control loops

## Objectives

- Predict CPU and workload demand using sliding-window ML forecasting
- Proactively scale resources before overload to reduce SLA violations
- Compare reactive vs. predictive schedulers on: overload rate, avg CPU%, avg capacity, total cost
- Persist trained model artifacts (`models/gbr_model.pkl`, `models/scaler.pkl`) for reuse

## References

- IEEE Papers on Cloud Scheduling and Predictive Autoscaling
- Gartner Reports on AIOps and Cloud Infrastructure Trends
- ACM Literature on Distributed Systems and Resource Management
