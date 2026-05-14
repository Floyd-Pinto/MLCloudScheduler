"""
model/train_all.py
-------------------
Master training pipeline for all Phase 2 models.

Execution order:
  1. GBR   (baseline)    → gbr_model.pkl, scaler.pkl
  2. LSTM  (attention)   → lstm_model.pt, lstm_scaler.pkl
  3. ARIMA (per-channel) → arima_meta.json, arima_history.pkl
  4. Combined (ensemble) → combined_meta.json
  5. AnomalyDetector     → anomaly_iforest.pkl

Usage:
    python model/train_all.py
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("MPLBACKEND", "Agg")

from model.workload_generator import generate_multivariate

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def train_all(verbose: bool = True) -> dict:
    """
    Train all models sequentially and return their metrics.

    Returns
    -------
    dict
        Keys: gbr, lstm, arima, combined, anomaly — each mapping to
        the model's per-resource metrics dict.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if os.environ.get("SKIP_TRAINING", "0").lower() in ("1", "true"):
        if verbose:
            print("============================================================")
            print("  SKIP_TRAINING is set to true. Bypassing model training.")
            print("============================================================")
        return {}

    results = {}
    t0 = time.time()

    # ── Generate diverse multivariate training corpus ─────────────────────────
    if verbose:
        print("=" * 60)
        print("  ML-Based Cloud Resource Scheduling — Phase 2 Training")
        print("  3-Resource Models: CPU, Memory, Network I/O")
        print("=" * 60)
        print("\n▶ Generating multi-resource training data...")

    data_full = generate_multivariate("combined", steps=600, seed=42)
    data_arima = generate_multivariate("combined", steps=300, seed=42)

    if verbose:
        print(f"  Generated {data_full.shape[0]} steps × {data_full.shape[1]} features\n")

    # ── 1. GBR ────────────────────────────────────────────────────────────────
    if verbose:
        print("─" * 40)
        print("  [1/5] Training GBR (MultiOutputRegressor)")
        print("─" * 40)
    try:
        from model.train_gbr import train as train_gbr
        results["gbr"] = train_gbr(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  ✗ GBR training failed: {e}")
        results["gbr"] = {"error": str(e)}

    # ── 2. LSTM ───────────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "─" * 40)
        print("  [2/5] Training LSTM (Multi-Resource)")
        print("─" * 40)
    try:
        from model.train_lstm import train as train_lstm
        results["lstm"] = train_lstm(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  ✗ LSTM training failed: {e}")
        results["lstm"] = {"error": str(e)}

    # ── 3. ARIMA ──────────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "─" * 40)
        print("  [3/5] Training ARIMA (Per-Channel AIC Grid Search)")
        print("─" * 40)
    try:
        from model.train_arima import train as train_arima
        results["arima"] = train_arima(data=data_arima, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  ✗ ARIMA training failed: {e}")
        results["arima"] = {"error": str(e)}

    # ── 4. Combined ───────────────────────────────────────────────────────────
    if verbose:
        print("\n" + "─" * 40)
        print("  [4/5] Training Combined Ensemble (LSTM + ARIMA)")
        print("─" * 40)
    try:
        from model.combined_model import CombinedForecaster
        combined = CombinedForecaster()
        lstm_m  = results.get("lstm", {})
        arima_m = results.get("arima", {})
        # Only pass metrics if both trained successfully
        if "error" not in lstm_m and "error" not in arima_m:
            results["combined"] = combined.fit(
                data_full, verbose=verbose,
                lstm_metrics=lstm_m, arima_metrics=arima_m
            )
        else:
            results["combined"] = combined.fit(data_full, verbose=verbose)
        combined.save()
    except Exception as e:
        if verbose:
            print(f"  ✗ Combined training failed: {e}")
        results["combined"] = {"error": str(e)}

    # ── 5. Anomaly Detector ───────────────────────────────────────────────────
    if verbose:
        print("\n" + "─" * 40)
        print("  [5/5] Fitting AnomalyDetector (IsolationForest + Z-Score)")
        print("─" * 40)
    try:
        from model.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        results["anomaly"] = det.fit(data_full, verbose=verbose)
        det.save()
    except Exception as e:
        if verbose:
            print(f"  ✗ AnomalyDetector fitting failed: {e}")
        results["anomaly"] = {"error": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    if verbose:
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print(f"  Total time: {elapsed:.1f}s")
        print("=" * 60)
        for model_name in ["gbr", "lstm", "arima", "combined"]:
            m = results.get(model_name, {})
            if "error" in m:
                print(f"  {model_name.upper():10s}  ✗  {m['error']}")
            else:
                r2 = m.get("r2", "—")
                rmse = m.get("rmse", "—")
                r2_str = f"{r2:.4f}" if isinstance(r2, float) else r2
                rmse_str = f"{rmse:.4f}" if isinstance(rmse, float) else rmse
                print(f"  {model_name.upper():10s}  R²={r2_str}  RMSE={rmse_str}")
        det_res = results.get("anomaly", {})
        if "error" not in det_res:
            print(f"  {'ANOMALY':10s}  "
                  f"IsolationForest fitted ({det_res.get('n_anomalies_detected', '?')} "
                  f"training anomalies)")
        print()

    # Invalidate inference cache
    try:
        from model.inference import invalidate_cache
        invalidate_cache()
    except Exception:
        pass

    # Save overall summary
    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    summary = {}
    for k, v in results.items():
        if isinstance(v, dict):
            # Filter out non-serializable values
            clean = {}
            for mk, mv in v.items():
                if isinstance(mv, (int, float, str, bool)):
                    clean[mk] = mv
                elif isinstance(mv, dict):
                    clean[mk] = {dk: dv for dk, dv in mv.items()
                                 if isinstance(dv, (int, float, str, bool))}
            summary[k] = clean
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return results


if __name__ == "__main__":
    train_all()
