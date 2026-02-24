# ML-Based Adaptive Cloud Resource Scheduling

A simulated cloud resource scheduling system that uses machine learning to **proactively** scale resources before overload occurs, compared against a traditional reactive (threshold-based) scheduler.

## Project Structure

```
ml-cloud-scheduler/
├── main.py                     # Entry point — runs the full simulation & comparison
├── workload_generator.py       # Synthetic workload patterns (gradual, spike, periodic)
├── schedulers/
│   ├── reactive_scheduler.py   # Baseline: threshold-based reactive scaling
│   └── predictive_scheduler.py # Proposed: ML-based predictive scaling
├── metrics/
│   └── collector.py            # Records CPU usage, overload events, cost estimates
├── visualization/
│   └── plotter.py              # Plots comparison graphs
├── data/                       # Generated simulation data (CSV)
├── models/                     # Saved ML model artifacts
├── requirements.txt
└── .gitignore
```

## Scheduling Approaches

| Approach | Mechanism | Scaling Trigger |
|---|---|---|
| **Reactive (Baseline)** | Static CPU thresholds | After overload detected |
| **Predictive (Proposed)** | ML time-series forecast | Before overload occurs |

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the simulation
```bash
python main.py
```

Results and plots are saved in the `data/` and `outputs/` directories.

## Technologies Used
- **Machine Learning** — Time-series forecasting (scikit-learn / statsmodels)
- **Simulation** — Synthetic workload generation
- **Visualization** — Matplotlib / Seaborn comparison plots
- **Concepts** — OS scheduling, cloud autoscaling, feedback control loops

## Objectives
- Predict CPU and workload demand using ML
- Proactively scale resources before overload
- Compare reactive vs. predictive schedulers on: overload rate, utilization, cost

## References
- IEEE Papers on Cloud Scheduling and Predictive Autoscaling
- Gartner Reports on AIOps and Cloud Infrastructure Trends
- ACM Literature on Distributed Systems and Resource Management
