"""
model/train_all.py
-------------------
Master training script — trains ALL 4 models and prints a comparison table.

Usage:
    python model/train_all.py

Outputs (all in model/saved_models/):
    gbr_model.pkl + scaler.pkl          (GBR)
    lstm_model.pt + lstm_scaler.pkl     (LSTM)
    arima_meta.json                     (ARIMA)
    combined_meta.json                  (Combined weights)
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.workload_generator import generate_gradual, generate_spike, generate_periodic, generate_combined


def build_full_series(seed=42) -> np.ndarray:
    """Combined training series from all patterns — diverse enough for LSTM."""
    s = np.concatenate([
        generate_gradual(steps=500, seed=seed),
        generate_spike(steps=500, seed=seed+1),
        generate_periodic(steps=500, seed=seed+2),
        generate_combined(steps=600, seed=seed+3),
        generate_gradual(steps=400, seed=seed+4),
        generate_spike(steps=400, spike_at=200, seed=seed+5),
        # Extra diversity for better LSTM generalisation
        generate_periodic(steps=400, seed=seed+6),
        generate_combined(steps=500, seed=seed+7),
        generate_spike(steps=300, spike_at=100, seed=seed+8),
        generate_gradual(steps=400, seed=seed+9),
    ])
    return s


def train_gbr(series: np.ndarray) -> dict:
    t0 = time.time()
    from model.train_gbr import train
    metrics = train()
    metrics["training_time"] = round(time.time() - t0, 2)
    return metrics


def train_lstm(series: np.ndarray) -> dict:
    t0 = time.time()
    from model.lstm_model import LSTMForecaster
    m = LSTMForecaster()
    metrics = m.fit(series, verbose=True)
    m.save()
    metrics["training_time"] = round(time.time() - t0, 2)
    return metrics


def train_arima(series: np.ndarray) -> dict:
    """ARIMA trains on a smaller chunk (faster walk-forward)."""
    t0 = time.time()
    from model.arima_model import ARIMAForecaster
    # Use a representative 300-step slice to keep training time manageable
    segment = series[:300]
    m = ARIMAForecaster()
    metrics = m.fit(segment, verbose=True)
    m.save()
    metrics["training_time"] = round(time.time() - t0, 2)
    return metrics


def train_combined(series: np.ndarray) -> dict:
    t0 = time.time()
    from model.combined_model import CombinedForecaster
    segment = series[:400]
    m = CombinedForecaster()
    metrics = m.fit(segment, verbose=True)
    m.save()
    metrics["training_time"] = round(time.time() - t0, 2)
    return metrics


def main():
    print("=" * 60)
    print("  ML CLOUD SCHEDULER — Training All Models")
    print("=" * 60)

    series = build_full_series()
    print(f"\nTraining data: {len(series)} time steps\n")

    results = {}

    print("\n[1/4] GradientBoosting Regressor (GBR)")
    print("-" * 40)
    results["GBR"] = train_gbr(series)

    print("\n[2/4] Long Short-Term Memory (LSTM) — PyTorch")
    print("-" * 40)
    results["LSTM"] = train_lstm(series)

    print("\n[3/4] ARIMA (Auto-Regressive Integrated Moving Average)")
    print("-" * 40)
    results["ARIMA"] = train_arima(series)

    print("\n[4/4] Combined Hybrid (LSTM + ARIMA)")
    print("-" * 40)
    results["Combined"] = train_combined(series)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    header = f"{'Model':<12} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Time(s)':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<12} {m.get('r2',0):>8.4f} {m.get('rmse',0):>8.4f} "
              f"{m.get('mae',0):>8.4f} {m.get('training_time',0):>10.1f}")

    best = max(results.items(), key=lambda x: x[1].get("r2", 0))
    print(f"\n✅ Best model: {best[0]}  (R²={best[1]['r2']:.4f})")

    return results


if __name__ == "__main__":
    main()
