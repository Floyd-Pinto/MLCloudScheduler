"""
model/evaluate.py
-----------------
Runs both schedulers over all workload patterns and returns
structured comparison results. Used by the backend's evaluation API.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.workload_generator import generate_gradual, generate_spike, generate_periodic, generate_combined, PATTERN_GENERATORS
from model.reactive_scheduler   import ReactiveScheduler
from model.predictive_scheduler import PredictiveScheduler
from model.metrics_collector    import MetricsCollector


def run_simulation(workload: np.ndarray, scheduler, collector: MetricsCollector,
                   scheduler_type: str) -> dict:
    """Step through the workload, collect metrics, return summary."""
    scheduler.reset()
    collector.reset()

    for t, load in enumerate(workload):
        if scheduler_type == "reactive":
            capacity = scheduler.decide(load)
        else:
            scheduler.observe(load)
            capacity = scheduler.decide()
        collector.record(t, float(load), capacity)

    return collector.summary()


def compare_schedulers(pattern: str = "combined", steps: int = 200,
                       seed: int = 42) -> dict:
    """
    Run reactive and predictive schedulers on the same workload pattern.
    Returns a dict with per-step data lists + summary stats for both.
    """
    generator = PATTERN_GENERATORS.get(pattern)
    if generator is None:
        raise ValueError(f"Unknown pattern '{pattern}'")

    if pattern == "combined":
        workload = generator(steps=steps, seed=seed)
    else:
        workload = generator(steps=steps, seed=seed)

    reactive_sched   = ReactiveScheduler(scale_up_threshold=70.0, scale_down_threshold=30.0, cooldown_steps=5)
    predictive_sched = PredictiveScheduler(window_size=10, horizon=5, scale_up_threshold=65.0,
                                           scale_down_threshold=30.0, cooldown_steps=3, retrain_every=10)

    r_collector = MetricsCollector(overload_threshold=80.0, scheduler_type="reactive")
    p_collector = MetricsCollector(overload_threshold=80.0, scheduler_type="predictive")

    r_summary = run_simulation(workload, reactive_sched,   r_collector,   "reactive")
    p_summary = run_simulation(workload, predictive_sched, p_collector, "predictive")

    r_df = r_collector.to_dataframe()
    p_df = p_collector.to_dataframe()

    def df_to_records(df):
        return df.to_dict(orient="records")

    return {
        "pattern":    pattern,
        "steps":      steps,
        "seed":       seed,
        "reactive": {
            "summary": r_summary,
            "records": df_to_records(r_df),
        },
        "predictive": {
            "summary": p_summary,
            "records": df_to_records(p_df),
        },
    }


if __name__ == "__main__":
    for pat in ["gradual", "spike", "periodic", "combined"]:
        result = compare_schedulers(pattern=pat)
        rs = result["reactive"]["summary"]
        ps = result["predictive"]["summary"]
        print(f"\n[{pat.upper()}]")
        print(f"  Reactive   — overloads: {rs['overload_events']:3d}  avg_cpu: {rs['avg_cpu']:.1f}%  cost: {rs['total_cost']:.0f}")
        print(f"  Predictive — overloads: {ps['overload_events']:3d}  avg_cpu: {ps['avg_cpu']:.1f}%  cost: {ps['total_cost']:.0f}")
