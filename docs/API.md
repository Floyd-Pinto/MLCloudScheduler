# API Reference

Base URL: `http://localhost:8000/api`

All endpoints accept and return JSON. No authentication required in development.

---

## Simulation — `/api/simulation/`

### POST /simulation/generate/

Generate a synthetic cloud workload pattern.

**Request:**
```json
{
  "pattern": "combined",   // gradual | spike | periodic | combined
  "steps": 200,            // 50–1000 (number of time points)
  "seed": 42,              // 0–99999 (random seed for reproducibility)
  "label": "My test"       // optional descriptive label
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
    {"time_step": 1, "workload": 22.31},
    ...
  ]
}
```

### GET /simulation/runs/
List all workload runs (summary only, no datapoints).

### GET /simulation/runs/{id}/
Get a specific run including all datapoints.

### DELETE /simulation/runs/{id}/
Delete a workload run and its datapoints.

---

## ML Model — `/api/ml/`

### POST /ml/train/

Train a specific model or all models. Training is synchronous.

**Request:**
```json
{"model_type": "lstm"}   // lstm | arima | combined | gbr | all
```

**Response:** `201 Created`
```json
{
  "id": 5,
  "model_type": "lstm",
  "status": "completed",
  "r2": 0.9696,
  "rmse": 5.1057,
  "mae": 4.0363,
  "extra_info": {},
  "started_at": "2026-04-27T04:12:22Z",
  "finished_at": "2026-04-27T04:12:52Z"
}
```

When `model_type: "all"`, trains GBR → LSTM → ARIMA → Combined sequentially (~2 minutes total).

### GET /ml/status/

Get training status and metrics for all models.

**Response:**
```json
{
  "statuses": {
    "gbr": true,
    "lstm": true,
    "arima": true,
    "combined": true
  },
  "gbr":      {"r2": 0.9709, "rmse": 4.8521, "mae": 3.3398, "finished_at": "..."},
  "lstm":     {"r2": 0.9696, "rmse": 5.1057, "mae": 4.0363, "finished_at": "..."},
  "arima":    {"r2": 0.6351, "rmse": 1.8506, "mae": 0.9267, "finished_at": "..."},
  "combined": {"r2": 0.7952, "rmse": 14.528, "mae": 6.5618, "extra_info": {"w_lstm": 0.434, "w_arima": 0.566}, "finished_at": "..."}
}
```

### POST /ml/predict/

Run inference on a history window.

**Request:**
```json
{
  "history": [30.5, 32.1, 35.0, 37.2, ...],   // at least 10 values
  "model_type": "lstm"                          // lstm | arima | combined | gbr
}
```

**Response:**
```json
{"prediction": 41.23, "model_type": "lstm"}
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
  "statuses": {"gbr": true, "lstm": true, "arima": true, "combined": true},
  "predictions": {"gbr": 41.23, "lstm": 39.81, "arima": 40.11, "combined": 40.05}
}
```

### POST /ml/compare-models/

Evaluate all models on a given workload pattern. Returns metrics + chart data.

**Request:**
```json
{"pattern": "spike", "steps": 250, "seed": 42}
```

**Response:**
```json
{
  "metrics": {
    "lstm":     {"r2": 0.9696, "rmse": 5.11, "mae": 4.04, "ready": true},
    "arima":    {"r2": 0.6351, "rmse": 1.85, "mae": 0.93, "ready": true},
    "combined": {"r2": 0.7952, "rmse": 14.5, "mae": 6.56, "ready": true}
  },
  "best_model": "lstm",
  "chart": {
    "timestamps": [0, 1, 2, ...],
    "actual": [20.7, 22.3, ...],
    "lstm": [21.1, 22.0, ...],
    "arima": [20.5, 21.8, ...],
    "combined": [20.8, 21.9, ...]
  }
}
```

### GET /ml/history/

List all training run records. Optional filter: `?model_type=lstm`.

---

## Scheduler — `/api/scheduler/`

### POST /scheduler/reactive/

Run the reactive (threshold-based) scheduler on a workload.

**Request:**
```json
{"pattern": "spike", "steps": 200, "seed": 42}
```

**Response:** `201 Created` — Full SchedulerRun object with `actions[]` array.

### POST /scheduler/predictive/

Run the ML-based predictive scheduler.

### POST /scheduler/compare/

Run **both** schedulers on the same workload and return side-by-side results.

**Response:**
```json
{
  "reactive": {
    "id": 33,
    "scheduler_type": "reactive",
    "pattern": "spike",
    "steps": 200,
    "overload_events": 31,
    "overload_rate": 15.5,
    "avg_cpu": 56.1,
    "avg_capacity": 6.8,
    "total_cost": 1358,
    "scale_up_count": 8,
    "scale_down_count": 2,
    "actions": [
      {"time_step": 0, "workload": 20.7, "capacity": 5, "cpu_usage": 41.4, "action": "hold", "overloaded": false},
      ...
    ]
  },
  "predictive": {
    "id": 34,
    "scheduler_type": "predictive",
    "overload_events": 19,
    "overload_rate": 9.5,
    ...
  }
}
```

### GET /scheduler/runs/
List all scheduler runs (summary, no actions).

### GET /scheduler/runs/{id}/
Get full run including step-by-step `actions[]`.

---

## Metrics — `/api/metrics/`

### GET /metrics/

List all scheduler run summaries. Filter with query params:
- `?type=reactive` or `?type=predictive`
- `?pattern=spike`

### GET /metrics/summary/

Aggregated KPIs across all runs.

**Response:**
```json
{
  "overall":    {"total_runs": 34, "avg_overload": 12.9, "avg_cpu": 59.8},
  "reactive":   {"total_runs": 17, "avg_overload": 14.7, "avg_cpu": 62.2, "run_count": 17},
  "predictive": {"total_runs": 17, "avg_overload": 11.1, "avg_cpu": 57.5, "run_count": 17}
}
```

---

## Evaluation — `/api/evaluation/`

### POST /evaluation/run/

Run both schedulers and save a formal comparison result.

**Request:**
```json
{"pattern": "spike", "steps": 200, "seed": 42}
```

**Response:** `201 Created`
```json
{
  "id": 1,
  "pattern": "spike",
  "steps": 200,
  "seed": 42,
  "r_overload_events": 31,
  "p_overload_events": 19,
  "r_overload_rate": 15.5,
  "p_overload_rate": 9.5,
  "overload_reduction": 38.7,
  "cost_difference": 261.0,
  "created_at": "2026-04-27T..."
}
```

### GET /evaluation/
List all saved evaluation results.

### GET /evaluation/comparison/?pattern=combined
Get the latest saved comparison for a specific pattern.
