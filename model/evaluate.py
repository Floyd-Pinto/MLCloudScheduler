"""
model/evaluate.py
-----------------
Runs both schedulers over all workload patterns and returns
structured comparison results. Used by the backend's evaluation API.

Updated for Phase 2: multi-resource scheduling with CPU, memory,
and network I/O signals.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.workload_generator import (
    generate, generate_multivariate, PATTERN_GENERATORS,
)
from model.reactive_scheduler   import ReactiveScheduler
from model.predictive_scheduler import PredictiveScheduler
from model.metrics_collector    import MetricsCollector, CAPACITY_PER_UNIT


def run_simulation(workload_mv: np.ndarray, scheduler, collector: MetricsCollector,
                   scheduler_type: str) -> dict:
    """
    Step through a multi-resource workload, collect metrics, return summary.

    Parameters
    ----------
    workload_mv : np.ndarray
        Shape (steps, 3) — [cpu_usage, memory_usage, network_io].
    """
    scheduler.reset()
    collector.reset()

    for t in range(len(workload_mv)):
        load = float(workload_mv[t, 0])  # CPU is the primary signal
        cpu_raw = float(workload_mv[t, 0])
        mem_raw = float(workload_mv[t, 1])
        net_raw = float(workload_mv[t, 2])

        if scheduler_type == "reactive":
            # Derive utilisation from raw workload and current capacity
            cap_units = max(scheduler.capacity * CAPACITY_PER_UNIT, 1)
            cpu_pct = min(cpu_raw / cap_units * 100.0, 100.0)
            mem_pct = min(mem_raw / cap_units * 100.0, 100.0)
            net_pct = min(net_raw / cap_units * 100.0, 100.0)
            capacity = scheduler.decide(load, cpu_pct, mem_pct, net_pct)
            trigger = scheduler.last_trigger
        else:
            # Predictive: observe, then decide
            cap_units = max(scheduler.capacity * CAPACITY_PER_UNIT, 1)
            cpu_pct = min(cpu_raw / cap_units * 100.0, 100.0)
            mem_pct = min(mem_raw / cap_units * 100.0, 100.0)
            net_pct = min(net_raw / cap_units * 100.0, 100.0)
            scheduler.observe(load, cpu_pct, mem_pct, net_pct)
            capacity = scheduler.decide()
            trigger = scheduler.last_trigger

        collector.record(
            time_step=t,
            workload=load,
            capacity=capacity,
            cpu_usage=cpu_pct,
            memory_usage=mem_pct,
            network_io=net_pct,
            trigger_resource=trigger,
        )

    return collector.summary()


def compare_schedulers(pattern: str = "combined", steps: int = 200,
                       seed: int = 42) -> dict:
    """
    Run reactive and predictive schedulers on the same multi-resource workload.
    """
    if pattern not in PATTERN_GENERATORS:
        raise ValueError(f"Unknown pattern '{pattern}'")

    workload_mv = generate_multivariate(pattern=pattern, steps=steps, seed=seed)

    reactive_sched   = ReactiveScheduler()
    predictive_sched = PredictiveScheduler(window_size=20, horizon=5)

    r_collector = MetricsCollector(scheduler_type="reactive")
    p_collector = MetricsCollector(scheduler_type="predictive")

    r_summary = run_simulation(workload_mv, reactive_sched,   r_collector,   "reactive")
    p_summary = run_simulation(workload_mv, predictive_sched, p_collector, "predictive")

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
