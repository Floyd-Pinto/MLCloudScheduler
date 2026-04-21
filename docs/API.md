# API Documentation

Base URL: `http://localhost:8000/api`

All endpoints accept and return JSON. No authentication required in development.

---

## Simulation

### POST /simulation/generate/
Generate a synthetic workload pattern.

**Request:**
```json
{
  "pattern": "combined",   // gradual | spike | periodic | combined
  "steps": 200,            // 50–1000
  "seed": 42,              // 0–99999
  "label": "My test"       // optional string
}
```

**Response:** `201 Created`
```json
{
  "id": 1,
  "pattern": "combined",
  "steps": 200,
  "seed": 42,
  "label": "My test",
  "created_at": "2026-04-21T...",
  "datapoints": [
    {"time_step": 0, "workload": 20.76},
    ...
  ]
}
```

### GET /simulation/runs/
List all workload runs (without datapoints).

### GET /simulation/runs/{id}/
Get a specific run including all datapoints.

### DELETE /simulation/runs/{id}/
Delete a workload run.

---

## Scheduler

### POST /scheduler/reactive/
Run the reactive (threshold-based) scheduler on a generated workload.

**Request:**
```json
{"pattern": "spike", "steps": 200, "seed": 42}
```

**Response:** `201 Created` — Full run object with `actions[]` array.

### POST /scheduler/predictive/
Run the ML predictive scheduler.

### POST /scheduler/compare/
Run **both** schedulers on the same workload and return both results side-by-side.

**Response:**
```json
{
  "reactive": {
    "id": 1,
    "scheduler_type": "reactive",
    "overload_events": 24,
    "overload_rate": 24.0,
    "avg_cpu": 70.47,
    "avg_capacity": 4.97,
    "total_cost": 497.0,
    "scale_up_count": 6,
    "scale_down_count": 0,
    "actions": [...]
  },
  "predictive": { ... }
}
```

### GET /scheduler/runs/
List all scheduler runs (summary only, no actions array).

### GET /scheduler/runs/{id}/
Get full run including step-by-step `actions[]`.

---

## ML Model

### POST /ml/train/
Trigger model training (synchronous, ~15–30s for GBR, up to 90s for all).

**Request:**
```json
{"model_type": "gbr"}   // gbr | lstm | arima | combined | all
```

**Response:** `201 Created`
```json
{
  "id": 2,
  "model_type": "gbr",
  "status": "completed",
  "rmse": 4.8521,
  "mae": 3.3398,
  "r2": 0.9709,
  "started_at": "...",
  "finished_at": "..."
}
```

### GET /ml/status/
Check if model is ready and get latest training metrics.

```json
{
  "gbr_ready": true,
  "latest_run": {
    "status": "completed",
    "r2": 0.9709,
    "rmse": 4.8521,
    "mae": 3.3398,
    "finished_at": "..."
  }
}
```

### POST /ml/predict/
Run inference on a history window.

**Request:**
```json
{
  "history": [30.5, 32.1, 35.0, ...],   // at least 10 values
  "model_type": "gbr"
}
```

**Response:**
```json
{"prediction": 41.23, "model_type": "gbr"}
```

### POST /ml/predict-all/
Run inference across all available models simultaneously.

**Request:**
```json
{"history": [30.5, 32.1, 35.0, ...]}
```

**Response:**
```json
{
  "statuses": { "gbr": true, "lstm": true, "arima": true, "combined": true },
  "predictions": { "gbr": 41.23, "lstm": 39.81, "arima": 40.11, "combined": 40.05 }
}
```

### POST /ml/compare-models/
Evaluate all 4 models on a given workload pattern and return metrics and chart data.

**Request:**
```json
{"pattern": "spike", "steps": 250, "seed": 42}
```

### GET /ml/history/
List all training run records. Filter optionally with `?model_type=gbr`.

---

## Metrics

### GET /metrics/
List all scheduler run summaries. Filter with query params:
- `?type=reactive` or `?type=predictive`
- `?pattern=spike`

### GET /metrics/summary/
Aggregate KPIs across all runs.

```json
{
  "overall":    {"total_runs": 10, "avg_overload": 18.2, "avg_cpu": 57.3},
  "reactive":   {"total_runs": 5, "avg_overload": 22.1, "avg_cpu": 60.1},
  "predictive": {"total_runs": 5, "avg_overload": 14.3, "avg_cpu": 54.5}
}
```

---

## Evaluation

### POST /evaluation/run/
Run both schedulers and save a formal comparison result.

**Request:** Same as scheduler endpoints — `pattern`, `steps`, `seed`.

**Response:** `201 Created`
```json
{
  "id": 1,
  "pattern": "spike",
  "overload_reduction": 9.5,
  "cost_difference": 59.0,
  "r_overload_events": 24,
  "p_overload_events": 19,
  ...
}
```

### GET /evaluation/
List all saved evaluation results.

### GET /evaluation/comparison/?pattern=combined
Get the latest saved comparison for a pattern.
