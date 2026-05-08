"""backend/ml_model/services.py — Updated for Phase 2 multi-resource models."""

import sys, os, json
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging
import numpy as np
from .models import ModelTrainingRun, ModelComparisonResult

logger = logging.getLogger(__name__)


def _build_multivariate_data(steps: int = 600, seed: int = 42) -> np.ndarray:
    """Generate multivariate training data for per-model training via API."""
    from model.workload_generator import generate_multivariate
    return generate_multivariate("combined", steps=steps, seed=seed)


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


# ── Model Comparison ──────────────────────────────────────────────────────────
def compare_all_models(pattern: str = "combined", steps: int = 300, seed: int = 42) -> dict:
    """
    Evaluate all 4 models: authoritative R²/RMSE/MAE come from the most recent
    training run records in the DB. Chart data is generated from multi-resource
    batch forward pass on the requested workload pattern.
    """
    from model.workload_generator import generate_multivariate
    from model.inference import predict, model_is_ready

    workload_mv = generate_multivariate(pattern=pattern, steps=max(steps, 200), seed=seed)

    # ── Pull metrics from DB training records (authoritative) ─────────────────
    metrics = {}
    for mt in ("gbr", "lstm", "arima", "combined"):
        latest = ModelTrainingRun.objects.filter(model_type=mt, status="completed").first()
        if latest and latest.r2 is not None:
            metrics[mt] = {
                "rmse":  round(latest.rmse, 4) if latest.rmse else None,
                "mae":   round(latest.mae,  4) if latest.mae  else None,
                "r2":    round(latest.r2,   4),
                "ready": True,
            }
            # Include per-resource metrics if available
            if latest.extra_info:
                for k, v in latest.extra_info.items():
                    if isinstance(v, (int, float)):
                        metrics[mt][k] = round(v, 4)
        else:
            metrics[mt] = {"rmse": None, "mae": None, "r2": None, "ready": False}

    # ── Generate chart data: CPU forecast vs actual ───────────────────────────
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
                    # Use CPU forecast (first step ahead with HORIZON offset)
                    cpu_forecast = result["cpu"]
                    forecasts[mt].append(float(cpu_forecast[-1]))  # last step of horizon
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
    best  = max(ready, key=lambda m: metrics[m].get("r2", -999)) if ready else ""

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
