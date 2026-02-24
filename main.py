"""
main.py
-------
Entry point for the ML-Based Adaptive Cloud Resource Scheduling simulation.

Runs both the Reactive and Predictive schedulers over each workload pattern,
collects metrics, saves CSVs, and produces comparison plots.
"""

import os
import pandas as pd

from workload_generator import (
    generate_gradual, generate_spike, generate_periodic, generate_combined,
)
from schedulers.reactive_scheduler import ReactiveScheduler
from schedulers.predictive_scheduler import PredictiveScheduler
from metrics.collector import MetricsCollector
from visualization.plotter import (
    plot_workload_vs_capacity, plot_cpu_usage, plot_summary_comparison,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

OVERLOAD_THRESHOLD = 80.0


# ---------------------------------------------------------------------------
def run_simulation(workload, scheduler, collector, scheduler_type: str):
    """
    Step through the workload array, letting the scheduler decide capacity
    at each step and the collector record metrics.
    """
    scheduler.reset()
    collector.reset()

    for t, load in enumerate(workload):
        if scheduler_type == "reactive":
            capacity = scheduler.decide(load)
        else:  # predictive
            scheduler.observe(load)
            capacity = scheduler.decide()

        collector.record(t, load, capacity)


# ---------------------------------------------------------------------------
def main():
    patterns = {
        "gradual":  generate_gradual(),
        "spike":    generate_spike(),
        "periodic": generate_periodic(),
        "combined": generate_combined(),
    }

    reactive_sched   = ReactiveScheduler(scale_up_threshold=70.0,
                                         scale_down_threshold=30.0,
                                         cooldown_steps=5)
    predictive_sched = PredictiveScheduler(window_size=10, horizon=5,
                                           scale_up_threshold=60.0,   # scales up earlier
                                           scale_down_threshold=30.0,
                                           cooldown_steps=3,           # reacts faster
                                           retrain_every=10)           # trains frequently

    reactive_collector   = MetricsCollector(overload_threshold=OVERLOAD_THRESHOLD)
    predictive_collector = MetricsCollector(overload_threshold=OVERLOAD_THRESHOLD)

    all_summaries = {}

    for pattern_name, workload in patterns.items():
        print(f"\n{'='*55}")
        print(f"  Pattern: {pattern_name.upper()}")
        print(f"{'='*55}")

        # --- Reactive ---
        run_simulation(workload, reactive_sched,   reactive_collector,   "reactive")
        # --- Predictive ---
        run_simulation(workload, predictive_sched, predictive_collector, "predictive")

        r_df = reactive_collector.to_dataframe()
        p_df = predictive_collector.to_dataframe()

        # Save CSVs
        r_df.to_csv(os.path.join(DATA_DIR, f"{pattern_name}_reactive.csv"),   index=False)
        p_df.to_csv(os.path.join(DATA_DIR, f"{pattern_name}_predictive.csv"), index=False)

        r_summary = reactive_collector.summary()
        p_summary = predictive_collector.summary()

        print("\n  [Reactive]")
        for k, v in r_summary.items():
            print(f"    {k:<22} {v}")
        print("\n  [Predictive]")
        for k, v in p_summary.items():
            print(f"    {k:<22} {v}")

        all_summaries[pattern_name] = {
            "Reactive":   r_summary,
            "Predictive": p_summary,
        }

        # Plots
        plot_workload_vs_capacity(r_df, p_df, pattern_name)
        plot_cpu_usage(r_df, p_df, pattern_name, OVERLOAD_THRESHOLD)
        plot_summary_comparison(
            {"Reactive": r_summary, "Predictive": p_summary}, pattern_name
        )

    print("\n\nSimulation complete. Outputs saved to outputs/")

    # Persist the trained ML model for later use / inspection
    predictive_sched.save_model("models")


if __name__ == "__main__":
    main()
