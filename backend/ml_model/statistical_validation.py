"""
backend/ml_model/statistical_validation.py
-------------------------------------------
Run N independent experiments to statistically validate LSTM superiority.

Each run generates a fresh workload with a unique seed, evaluates all 4 models
on that workload, and records per-model R²/RMSE/MAE. After all runs, computes
summary statistics including win-rates and pairwise comparisons.
"""

import sys
import os
import time
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def _evaluate_model_on_workload(workload_mv: np.ndarray, model_type: str,
                                  window_size: int = 20, horizon: int = 5) -> dict:
    """
    Evaluate a single model on a workload using sliding window.
    Returns {'r2': float, 'rmse': float, 'mae': float} or None-valued dict on failure.
    """
    from model.inference import predict, model_is_ready

    if not model_is_ready(model_type):
        return {"r2": None, "rmse": None, "mae": None}

    n = len(workload_mv)
    start = int(n * 0.2)
    end = int(n * 0.8)

    actuals = []
    preds = []

    # Use wider step for speed (every 10th position) to prevent HTTP timeouts
    for t in range(start + window_size, end - horizon, 10):
        actual_idx = t + horizon
        if actual_idx >= n:
            break
        try:
            window = workload_mv[:t + 1]
            result = predict(model_type=model_type, window=window)
            # Use all 3 resources for comprehensive evaluation
            for res_idx, key in enumerate(["cpu", "memory", "network"]):
                preds.append(float(result[key][-1]))
                actuals.append(float(workload_mv[actual_idx, res_idx]))
        except Exception:
            continue

    if len(actuals) < 10:
        return {"r2": None, "rmse": None, "mae": None}

    a = np.array(actuals)
    p = np.array(preds)

    return {
        "r2":   round(float(r2_score(a, p)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(a, p))), 4),
        "mae":  round(float(mean_absolute_error(a, p)), 4),
    }


def run_statistical_validation(n_runs: int = 20, pattern: str = "combined",
                                steps: int = 300) -> dict:
    """
    Run n_runs independent evaluations with different seeds to
    statistically validate model performance.

    Parameters
    ----------
    n_runs : int
        Number of independent experimental runs.
    pattern : str
        Workload pattern to use.
    steps : int
        Number of time steps per run.

    Returns
    -------
    dict
        Comprehensive results including per-run metrics, summary statistics,
        win-rates, and pairwise comparisons.
    """
    from model.workload_generator import generate_multivariate

    model_types = ["lstm", "gbr", "arima", "combined"]

    # Per-run results
    runs = []

    t0 = time.time()

    for i in range(n_runs):
        run_seed = i * 7 + 100  # Diverse seeds
        workload_mv = generate_multivariate(pattern=pattern, steps=steps, seed=run_seed)

        run_result = {"run": i + 1, "seed": run_seed}
        r2_vals = {}

        for mt in model_types:
            metrics = _evaluate_model_on_workload(workload_mv, mt)
            run_result[f"{mt}_r2"] = metrics["r2"]
            run_result[f"{mt}_rmse"] = metrics["rmse"]
            run_result[f"{mt}_mae"] = metrics["mae"]
            if metrics["r2"] is not None:
                r2_vals[mt] = metrics["r2"]

        # Determine winner for this run
        if r2_vals:
            winner = max(r2_vals, key=r2_vals.get)
            run_result["winner"] = winner
        else:
            run_result["winner"] = None

        runs.append(run_result)

    elapsed = time.time() - t0

    # ── Summary statistics ────────────────────────────────────────────────
    summary = {}
    for mt in model_types:
        r2_values = [r[f"{mt}_r2"] for r in runs if r.get(f"{mt}_r2") is not None]
        if r2_values:
            arr = np.array(r2_values)
            summary[mt] = {
                "mean_r2":   round(float(arr.mean()), 4),
                "std_r2":    round(float(arr.std()), 4),
                "min_r2":    round(float(arr.min()), 4),
                "max_r2":    round(float(arr.max()), 4),
                "median_r2": round(float(np.median(arr)), 4),
                "n_valid":   len(r2_values),
            }
        else:
            summary[mt] = {
                "mean_r2": None, "std_r2": None, "min_r2": None,
                "max_r2": None, "median_r2": None, "n_valid": 0,
            }

    # ── Win-rate analysis ─────────────────────────────────────────────────
    win_counts = {mt: 0 for mt in model_types}
    valid_runs = 0
    for r in runs:
        if r.get("winner"):
            win_counts[r["winner"]] += 1
            valid_runs += 1

    win_rates = {}
    for mt in model_types:
        win_rates[mt] = round(win_counts[mt] / max(valid_runs, 1), 4)

    # ── Pairwise comparisons (LSTM vs others) ────────────────────────────
    pairwise = {}
    for other in ["gbr", "arima", "combined"]:
        lstm_wins = 0
        diffs = []
        n_compare = 0
        for r in runs:
            lstm_r2 = r.get("lstm_r2")
            other_r2 = r.get(f"{other}_r2")
            if lstm_r2 is not None and other_r2 is not None:
                n_compare += 1
                diff = lstm_r2 - other_r2
                diffs.append(diff)
                if lstm_r2 > other_r2:
                    lstm_wins += 1

        pairwise[f"lstm_vs_{other}"] = {
            "lstm_wins": lstm_wins,
            "total_runs": n_compare,
            "lstm_win_pct": round(lstm_wins / max(n_compare, 1) * 100, 1),
            "mean_diff": round(float(np.mean(diffs)), 4) if diffs else None,
            "std_diff": round(float(np.std(diffs)), 4) if diffs else None,
        }

    return {
        "n_runs": n_runs,
        "pattern": pattern,
        "steps": steps,
        "elapsed_seconds": round(elapsed, 1),
        "runs": runs,
        "summary": summary,
        "win_counts": win_counts,
        "win_rates": win_rates,
        "lstm_win_rate": win_rates.get("lstm", 0),
        "pairwise": pairwise,
    }
