"""backend/ml_model/services.py — Updated for Phase 2 multi-resource models."""

import sys, os, json
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from .models import ModelTrainingRun, ModelComparisonResult

logger = logging.getLogger(__name__)


def _build_multivariate_data(steps: int = 600, seed: int = 42) -> np.ndarray:
    """Generate multivariate training data for per-model training via API."""
    from model.workload_generator import generate_multivariate
    return generate_multivariate("combined", steps=steps, seed=seed)


def _generate_workload(pattern: str, steps: int, seed: int) -> np.ndarray:
    """Generate workload data, supporting both synthetic and real-data patterns."""
    if pattern in ("google_trace", "alibaba_trace"):
        from model.workload_generator import generate_from_real_data
        source = "google" if pattern == "google_trace" else "alibaba"
        return generate_from_real_data(source=source, steps=steps, start_offset=seed)
    else:
        from model.workload_generator import generate_multivariate
        return generate_multivariate(pattern=pattern, steps=max(steps, 200), seed=seed)


# ── Training ──────────────────────────────────────────────────────────────────
def trigger_training(model_type: str = "gbr") -> ModelTrainingRun:
    record = ModelTrainingRun.objects.create(model_type=model_type, status="running")
    try:
        metrics = _run_training(model_type)
        if metrics:
            record.rmse      = float(metrics["rmse"])   if metrics.get("rmse")  is not None else None
            record.mae       = float(metrics["mae"])    if metrics.get("mae")   is not None else None
            record.r2        = float(metrics["r2"])     if metrics.get("r2")    is not None else None
            record.status    = "completed"
            # Store extra info (per-resource metrics, weights, etc.)
            extra = {}
            for k, v in metrics.items():
                if k not in ("rmse", "mae", "r2", "training_time"):
                    if isinstance(v, (int, float)):
                        extra[k] = float(v)
                    elif isinstance(v, dict):
                        extra[k] = {dk: float(dv) for dk, dv in v.items()
                                    if isinstance(dv, (int, float))}
            if extra:
                record.extra_info = extra
        else:
            record.status    = "failed"
            record.error_msg = "Training returned no metrics"
    except Exception as exc:
        logger.exception("Model training error")
        record.status    = "failed"
        record.error_msg = str(exc)[:500]
    finally:
        try:
            from model.inference import invalidate_cache
            invalidate_cache(model_type)
        except Exception:
            pass

    record.finished_at = datetime.now(tz=timezone.utc)
    record.save()
    return record


def trigger_train_all() -> list[ModelTrainingRun]:
    """Train all 4 models sequentially. Returns a list of run records."""
    records = []
    for mt in ("gbr", "lstm", "arima", "combined"):
        records.append(trigger_training(mt))
    return records


def _run_training(model_type: str) -> dict | None:
    if model_type == "gbr":
        from model.train_gbr import train
        return train()

    if model_type == "lstm":
        from model.train_lstm import train
        return train()

    if model_type == "arima":
        from model.train_arima import train
        return train()

    if model_type == "combined":
        from model.combined_model import CombinedForecaster
        data = _build_multivariate_data()
        m = CombinedForecaster()
        # Get per-resource metrics from latest training runs
        lstm_run  = ModelTrainingRun.objects.filter(model_type="lstm", status="completed").first()
        arima_run = ModelTrainingRun.objects.filter(model_type="arima", status="completed").first()

        lstm_metrics = None
        arima_metrics = None
        if lstm_run and lstm_run.r2 is not None:
            lstm_metrics = {"rmse": lstm_run.rmse, "mae": lstm_run.mae, "r2": lstm_run.r2}
            # Include per-resource metrics from extra_info
            if lstm_run.extra_info:
                lstm_metrics.update(lstm_run.extra_info)
        if arima_run and arima_run.r2 is not None:
            arima_metrics = {"rmse": arima_run.rmse, "mae": arima_run.mae, "r2": arima_run.r2}
            if arima_run.extra_info:
                arima_metrics.update(arima_run.extra_info)

        metrics = m.fit(data, verbose=False, lstm_metrics=lstm_metrics, arima_metrics=arima_metrics)
        m.save()
        return metrics

    raise ValueError(f"Unknown model_type: {model_type}")


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(history: list[float], model_type: str = "gbr") -> dict:
    from model.inference import predict, model_is_ready
    if not model_is_ready(model_type):
        return {"error": f"Model '{model_type}' not ready. Please train it first."}
    try:
        result = predict(model_type=model_type, window=np.array(history))
        return {"prediction": result, "model_type": model_type}
    except Exception as exc:
        return {"error": str(exc)}


def run_inference_all(history: list[float]) -> dict:
    """Run all available models. Returns dict with predictions + readiness."""
    from model.inference import predict_all, all_model_statuses
    statuses = all_model_statuses()
    preds = predict_all(np.array(history))
    return {"statuses": statuses, "predictions": preds}


# ── Status ─────────────────────────────────────────────────────────────────────
def get_model_status() -> dict:
    from model.inference import all_model_statuses
    statuses = all_model_statuses()

    result = {"statuses": statuses}
    for mt in ("gbr", "lstm", "arima", "combined"):
        latest = ModelTrainingRun.objects.filter(model_type=mt, status="completed").first()
        result[mt] = {
            "ready":       statuses.get(mt, False),
            "r2":          latest.r2          if latest else None,
            "rmse":        latest.rmse        if latest else None,
            "mae":         latest.mae         if latest else None,
            "extra_info":  latest.extra_info  if latest else {},
            "finished_at": latest.finished_at.isoformat() if (latest and latest.finished_at) else None,
        }
    return result


# ── Model Comparison (LIVE metrics on same workload) ─────────────────────────
def _compute_live_metrics(workload_mv: np.ndarray, model_type: str,
                          window_size: int = 20, horizon: int = 5) -> dict:
    """
    Compute LIVE R², RMSE, MAE for a model on the given workload using
    a proper sliding window evaluation. This ensures all models are
    evaluated on the EXACT same data — no stale DB values.

    Returns per-resource and overall metrics.
    """
    from model.inference import predict, model_is_ready

    if not model_is_ready(model_type):
        return {"r2": None, "rmse": None, "mae": None, "ready": False,
                "cpu_r2": None, "mem_r2": None, "net_r2": None}

    n = len(workload_mv)
    start = int(n * 0.2)
    end = int(n * 0.8)

    actuals_cpu = []
    actuals_mem = []
    actuals_net = []
    preds_cpu = []
    preds_mem = []
    preds_net = []

    # Sliding window evaluation: predict HORIZON steps ahead
    for t in range(start + window_size, end - horizon):
        window = workload_mv[:t + 1]
        actual_idx = t + horizon

        if actual_idx >= n:
            break

        try:
            result = predict(model_type=model_type, window=window)
            # result has 'cpu', 'memory', 'network' — each a list of HORIZON values
            preds_cpu.append(float(result["cpu"][-1]))
            preds_mem.append(float(result["memory"][-1]))
            preds_net.append(float(result["network"][-1]))

            actuals_cpu.append(float(workload_mv[actual_idx, 0]))
            actuals_mem.append(float(workload_mv[actual_idx, 1]))
            actuals_net.append(float(workload_mv[actual_idx, 2]))
        except Exception:
            continue

    if len(actuals_cpu) < 10:
        return {"r2": None, "rmse": None, "mae": None, "ready": True,
                "cpu_r2": None, "mem_r2": None, "net_r2": None}

    a_cpu, p_cpu = np.array(actuals_cpu), np.array(preds_cpu)
    a_mem, p_mem = np.array(actuals_mem), np.array(preds_mem)
    a_net, p_net = np.array(actuals_net), np.array(preds_net)

    # Per-resource R²
    cpu_r2 = float(r2_score(a_cpu, p_cpu))
    mem_r2 = float(r2_score(a_mem, p_mem))
    net_r2 = float(r2_score(a_net, p_net))

    cpu_rmse = float(np.sqrt(mean_squared_error(a_cpu, p_cpu)))
    mem_rmse = float(np.sqrt(mean_squared_error(a_mem, p_mem)))
    net_rmse = float(np.sqrt(mean_squared_error(a_net, p_net)))

    # Overall: combine all resources
    a_all = np.concatenate([a_cpu, a_mem, a_net])
    p_all = np.concatenate([p_cpu, p_mem, p_net])

    overall_r2 = float(r2_score(a_all, p_all))
    overall_rmse = float(np.sqrt(mean_squared_error(a_all, p_all)))
    overall_mae = float(mean_absolute_error(a_all, p_all))

    return {
        "r2": round(overall_r2, 4),
        "rmse": round(overall_rmse, 4),
        "mae": round(overall_mae, 4),
        "cpu_r2": round(cpu_r2, 4),
        "cpu_rmse": round(cpu_rmse, 4),
        "mem_r2": round(mem_r2, 4),
        "mem_rmse": round(mem_rmse, 4),
        "net_r2": round(net_r2, 4),
        "net_rmse": round(net_rmse, 4),
        "ready": True,
    }


def compare_all_models(pattern: str = "combined", steps: int = 300, seed: int = 42) -> dict:
    """
    Evaluate all 4 models on the SAME workload using LIVE inference.
    All metrics are computed on the fly — no stale DB values.
    """
    from model.inference import predict, model_is_ready

    workload_mv = _generate_workload(pattern, steps, seed)

    # ── Compute LIVE metrics for all 4 models on the SAME workload ───────
    metrics = {}
    for mt in ("gbr", "lstm", "arima", "combined"):
        metrics[mt] = _compute_live_metrics(workload_mv, mt)

    # ── Generate chart data: CPU forecast vs actual ───────────────────────
    WINDOW  = 20
    HORIZON = 5
    cpu_series = workload_mv[:, 0]
    start   = int(len(cpu_series) * 0.2)
    end     = int(len(cpu_series) * 0.8)
    test_ts = list(range(start + WINDOW, end - HORIZON))

    actuals    = [float(cpu_series[t + HORIZON]) for t in test_ts]
    timestamps = test_ts
    forecasts  = {mt: [] for mt in ("gbr", "lstm", "arima", "combined")}

    for t in test_ts:
        window = workload_mv[:t + 1]  # multi-resource window
        for mt in ("gbr", "lstm", "arima", "combined"):
            if model_is_ready(mt):
                try:
                    result = predict(model_type=mt, window=window)
                    cpu_forecast = result["cpu"]
                    forecasts[mt].append(float(cpu_forecast[-1]))
                except Exception:
                    forecasts[mt].append(None)
            else:
                forecasts[mt].append(None)

    # Sample ≤150 pts for the chart
    N   = len(actuals)
    idx = list(range(0, N, max(1, N // 150)))
    chart = {
        "timestamps": [timestamps[i] for i in idx],
        "actual":     [actuals[i]   for i in idx],
    }
    for mt in ("gbr", "lstm", "arima", "combined"):
        chart[mt] = [forecasts[mt][i] for i in idx]

    # Best model by R²
    ready = [mt for mt in metrics if metrics[mt].get("ready")]
    best  = max(ready, key=lambda m: metrics[m].get("r2") or -999) if ready else ""

    # Save to DB
    comp = ModelComparisonResult.objects.create(
        series_length=len(workload_mv), pattern=pattern, seed=seed, best_model=best,
        gbr_rmse=metrics["gbr"].get("rmse"),      gbr_mae=metrics["gbr"].get("mae"),      gbr_r2=metrics["gbr"].get("r2"),
        lstm_rmse=metrics["lstm"].get("rmse"),     lstm_mae=metrics["lstm"].get("mae"),     lstm_r2=metrics["lstm"].get("r2"),
        arima_rmse=metrics["arima"].get("rmse"),   arima_mae=metrics["arima"].get("mae"),   arima_r2=metrics["arima"].get("r2"),
        combined_rmse=metrics["combined"].get("rmse"), combined_mae=metrics["combined"].get("mae"),
        combined_r2=metrics["combined"].get("r2"),
    )

    return {"id": comp.id, "metrics": metrics, "chart": chart, "best_model": best}
