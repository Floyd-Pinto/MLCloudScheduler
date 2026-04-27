"""scheduler/services.py — runs the reactive/predictive schedulers."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from model.workload_generator       import generate
from model.reactive_scheduler       import ReactiveScheduler
from model.predictive_scheduler     import PredictiveScheduler
from model.metrics_collector        import MetricsCollector

from .models import SchedulerRun, SchedulerAction
from simulation.models import WorkloadRun


def _run_and_save(workload: np.ndarray, scheduler, collector: MetricsCollector,
                  scheduler_type: str, pattern: str, seed: int,
                  workload_run: WorkloadRun | None) -> SchedulerRun:
    scheduler.reset()
    collector.reset()

    for t, load in enumerate(workload):
        if scheduler_type == "reactive":
            capacity = scheduler.decide(float(load))
        else:
            scheduler.observe(float(load))
            capacity = scheduler.decide()
        collector.record(t, float(load), capacity)

    summary = collector.summary()
    run = SchedulerRun.objects.create(
        workload_run    = workload_run,
        scheduler_type  = scheduler_type,
        pattern         = pattern,
        steps           = len(workload),
        seed            = seed,
        overload_events  = summary["overload_events"],
        overload_rate    = summary["overload_rate"],
        avg_cpu          = summary["avg_cpu"],
        avg_capacity     = summary["avg_capacity"],
        total_cost       = summary["total_cost"],
        scale_up_count   = summary["scale_up_count"],
        scale_down_count = summary["scale_down_count"],
    )

    df = collector.to_dataframe()
    actions = [
        SchedulerAction(
            run       = run,
            time_step = int(row.time_step),
            workload  = float(row.workload),
            capacity  = int(row.capacity),
            cpu_usage = float(row.cpu_usage),
            overloaded= bool(row.overloaded),
            action    = row.action,
        )
        for row in df.itertuples()
    ]
    SchedulerAction.objects.bulk_create(actions)
    return run


def run_reactive(pattern: str, steps: int, seed: int,
                 workload_run: WorkloadRun | None = None) -> SchedulerRun:
    workload   = generate(pattern=pattern, steps=steps, seed=seed)
    scheduler  = ReactiveScheduler(scale_up_threshold=70.0, scale_down_threshold=30.0, cooldown_steps=5)
    collector  = MetricsCollector(overload_threshold=80.0, scheduler_type="reactive")
    return _run_and_save(workload, scheduler, collector, "reactive", pattern, seed, workload_run)


def run_predictive(pattern: str, steps: int, seed: int,
                   workload_run: WorkloadRun | None = None) -> SchedulerRun:
    workload   = generate(pattern=pattern, steps=steps, seed=seed)
    scheduler  = PredictiveScheduler(window_size=10, horizon=5, scale_up_threshold=55.0,
                                     scale_down_threshold=30.0, cooldown_steps=2, retrain_every=10)
    collector  = MetricsCollector(overload_threshold=80.0, scheduler_type="predictive")
    return _run_and_save(workload, scheduler, collector, "predictive", pattern, seed, workload_run)


def run_comparison(pattern: str, steps: int, seed: int) -> dict:
    """Run both schedulers on the same workload and return both results."""
    r_run = run_reactive(pattern=pattern, steps=steps, seed=seed)
    p_run = run_predictive(pattern=pattern, steps=steps, seed=seed)
    return {"reactive_id": r_run.pk, "predictive_id": p_run.pk}
