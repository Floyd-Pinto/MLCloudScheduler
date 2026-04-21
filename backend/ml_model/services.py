"""backend/ml_model/services.py — handles all 4 model types."""

import sys, os, json
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging
from .models import ModelTrainingRun, ModelComparisonResult

logger = logging.getLogger(__name__)


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
            # store extra info (e.g. w_lstm, w_arima for combined)
            extra = {k: float(v) for k, v in metrics.items()
                     if k not in ("rmse", "mae", "r2", "training_time") and isinstance(v, (int, float))}
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
        # Invalidate inference cache so new model is loaded on next predict
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
        import numpy as np
        from model.lstm_model import LSTMForecaster
        from model.train_all import build_full_series
        series = build_full_series()
        m = LSTMForecaster()
        metrics = m.fit(series, verbose=False)
        m.save()
        return metrics

    if model_type == "arima":
        import numpy as np
        from model.arima_model import ARIMAForecaster
        from model.train_all import build_full_series
        series = build_full_series()[:300]
        m = ARIMAForecaster()
        metrics = m.fit(series, verbose=False)
        m.save()
        return metrics

    if model_type == "combined":
        import numpy as np
        from model.combined_model import CombinedForecaster
        from model.train_all import build_full_series
        series = build_full_series()[:600]
        m = CombinedForecaster()
        metrics = m.fit(series, verbose=False)
        m.save()
        return metrics

    raise ValueError(f"Unknown model_type: {model_type}")


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(history: list[float], model_type: str = "gbr") -> dict:
    from model.inference import predict, model_is_ready
    if not model_is_ready(model_type):
        return {"error": f"Model '{model_type}' not ready. Please train it first."}
    try:
        value = predict(history, model_type=model_type)
        return {"prediction": round(value, 4), "model_type": model_type}
    except Exception as exc:
        return {"error": str(exc)}


def run_inference_all(history: list[float]) -> dict:
    """Run all available models. Returns dict with predictions + readiness."""
    from model.inference import predict_all, all_model_statuses
    statuses = all_model_statuses()
    preds = predict_all(history)
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
    training run records in the DB (the same evaluation used at training time).
    Chart data is generated by a fast batch forward pass on the requested pattern.
    """
    from model.workload_generator import generate_gradual, generate_spike, generate_periodic, generate_combined
    from model.inference import predict, model_is_ready
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    generators = {
        "gradual": generate_gradual, "spike": generate_spike,
        "periodic": generate_periodic, "combined": generate_combined,
    }
    gen    = generators.get(pattern, generate_combined)
    series = gen(steps=max(steps, 200), seed=seed)

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
        else:
            metrics[mt] = {"rmse": None, "mae": None, "r2": None, "ready": False}

    # ── Generate chart data: batch sliding-window forward pass ────────────────
    WINDOW  = 15
    HORIZON = 5
    # Use the middle 60% of the series to avoid edge effects
    start   = int(len(series) * 0.2)
    end     = int(len(series) * 0.8)
    test_ts = list(range(start + WINDOW, end - HORIZON))

    actuals    = [float(series[t + HORIZON]) for t in test_ts]
    timestamps = test_ts
    forecasts  = {mt: [] for mt in ("gbr", "lstm", "arima", "combined")}

    for t in test_ts:
        hist = list(series[: t + 1])
        for mt in ("gbr", "lstm", "arima", "combined"):
            if model_is_ready(mt):
                try:
                    forecasts[mt].append(float(predict(hist, model_type=mt)))
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
        series_length=len(series), pattern=pattern, seed=seed, best_model=best,
        gbr_rmse=metrics["gbr"].get("rmse"),      gbr_mae=metrics["gbr"].get("mae"),      gbr_r2=metrics["gbr"].get("r2"),
        lstm_rmse=metrics["lstm"].get("rmse"),     lstm_mae=metrics["lstm"].get("mae"),     lstm_r2=metrics["lstm"].get("r2"),
        arima_rmse=metrics["arima"].get("rmse"),   arima_mae=metrics["arima"].get("mae"),   arima_r2=metrics["arima"].get("r2"),
        combined_rmse=metrics["combined"].get("rmse"), combined_mae=metrics["combined"].get("mae"),
        combined_r2=metrics["combined"].get("r2"),
    )

    return {"id": comp.id, "metrics": metrics, "chart": chart, "best_model": best}


