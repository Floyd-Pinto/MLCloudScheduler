"""scheduler/services.py — Phase 2: multi-resource scheduling with anomaly logging."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from model.workload_generator       import generate_multivariate
from model.reactive_scheduler       import ReactiveScheduler
from model.predictive_scheduler     import PredictiveScheduler
from model.metrics_collector        import MetricsCollector, CAPACITY_PER_UNIT
from model.anomaly_detector         import AnomalyDetector

from .models import SchedulerRun, SchedulerAction
from simulation.models import WorkloadRun


def _get_anomaly_detector():
    """Load the trained anomaly detector, or return None if not available."""
    try:
        det = AnomalyDetector()
        if det.load():
            det.reset()
            return det
    except Exception:
        pass
    return None


def _run_and_save(workload_mv: np.ndarray, scheduler, collector: MetricsCollector,
                  scheduler_type: str, pattern: str, seed: int,
                  workload_run: WorkloadRun | None) -> SchedulerRun:
    """
    Run a multi-resource scheduler simulation and persist results.
    For predictive runs, also runs the anomaly detector and saves
    anomaly logs to the anomaly app's AnomalyLogEntry table.

    Parameters
    ----------
    workload_mv : np.ndarray
        Shape (steps, 3) — [cpu_usage, memory_usage, network_io].
    """
    scheduler.reset()
    collector.reset()

    # Load anomaly detector for predictive runs
    anomaly_det = None
    if scheduler_type == "predictive":
        anomaly_det = _get_anomaly_detector()

    anomaly_entries = []  # collect for bulk insert

    for t in range(len(workload_mv)):
        load = float(workload_mv[t, 0])
        cpu_raw = float(workload_mv[t, 0])
        mem_raw = float(workload_mv[t, 1])
        net_raw = float(workload_mv[t, 2])

        # Derive utilisation from raw workload and current capacity
        cap_units = max(scheduler.capacity * CAPACITY_PER_UNIT, 1)
        cpu_pct = min(cpu_raw / cap_units * 100.0, 100.0)
        mem_pct = min(mem_raw / cap_units * 100.0, 100.0)
        net_pct = min(net_raw / cap_units * 100.0, 100.0)

        # Run anomaly detection on each step (predictive only)
        is_anomaly = False
        z_flag = False
        iso_flag = False
        if anomaly_det is not None:
            obs = np.array([cpu_pct, mem_pct, net_pct])
            # Check z-score and iforest separately for logging
            anomaly_det._history.append(obs.tolist())
            if len(anomaly_det._history) >= anomaly_det.rolling_window:
                recent = np.array(anomaly_det._history[-anomaly_det.rolling_window:])
                means = recent.mean(axis=0)
                stds  = recent.std(axis=0) + 1e-8
                z_scores = np.abs((obs - means) / stds)
                z_flag = bool(np.any(z_scores > anomaly_det.z_threshold))
            if anomaly_det._iso_forest is not None:
                pred = anomaly_det._iso_forest.predict(obs.reshape(1, -1))
                iso_flag = bool(pred[0] == -1)
            is_anomaly = z_flag or iso_flag

            anomaly_entries.append({
                "time_step": t,
                "cpu_usage": cpu_pct,
                "memory_usage": mem_pct,
                "network_io": net_pct,
                "is_anomaly": is_anomaly,
                "z_score_flag": z_flag,
                "iforest_flag": iso_flag,
                "pattern": pattern,
                "scheduler_type": scheduler_type,
            })

        if scheduler_type == "reactive":
            capacity = scheduler.decide(load, cpu_pct, mem_pct, net_pct)
            trigger = scheduler.last_trigger
        else:
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

    summary = collector.summary()
    run = SchedulerRun.objects.create(
        workload_run    = workload_run,
        scheduler_type  = scheduler_type,
        pattern         = pattern,
        steps           = len(workload_mv),
        seed            = seed,
        overload_events  = summary["overload_events"],
        overload_rate    = summary["overload_rate"],
        avg_cpu          = summary["avg_cpu"],
        avg_memory       = summary.get("avg_memory", 0.0),
        avg_network      = summary.get("avg_network", 0.0),
        avg_capacity     = summary["avg_capacity"],
        total_cost       = summary["total_cost"],
        scale_up_count   = summary["scale_up_count"],
        scale_down_count = summary["scale_down_count"],
        overload_cpu_count     = summary.get("overload_cpu", 0),
        overload_memory_count  = summary.get("overload_memory", 0),
        overload_network_count = summary.get("overload_network", 0),
    )

    df = collector.to_dataframe()
    actions = [
        SchedulerAction(
            run              = run,
            time_step        = int(row.time_step),
            workload         = float(row.workload),
            capacity         = int(row.capacity),
            cpu_usage        = float(row.cpu_usage),
            memory_usage     = float(row.memory_usage),
            network_io       = float(row.network_io),
            overloaded       = bool(row.overload_any),
            action           = row.action,
            trigger_resource = str(row.trigger_resource),
        )
        for row in df.itertuples()
    ]
    SchedulerAction.objects.bulk_create(actions)

    # Persist anomaly log entries
    if anomaly_entries:
        from anomaly.models import AnomalyLogEntry
        AnomalyLogEntry.objects.bulk_create([
            AnomalyLogEntry(**entry) for entry in anomaly_entries
        ])

    return run


def run_reactive(pattern: str, steps: int, seed: int,
                 workload_run: WorkloadRun | None = None) -> SchedulerRun:
    workload_mv = generate_multivariate(pattern=pattern, steps=steps, seed=seed)
    scheduler   = ReactiveScheduler()
    collector   = MetricsCollector(scheduler_type="reactive")
    return _run_and_save(workload_mv, scheduler, collector, "reactive", pattern, seed, workload_run)


def run_predictive(pattern: str, steps: int, seed: int,
                   workload_run: WorkloadRun | None = None) -> SchedulerRun:
    workload_mv = generate_multivariate(pattern=pattern, steps=steps, seed=seed)
    scheduler   = PredictiveScheduler(window_size=20, horizon=5)
    collector   = MetricsCollector(scheduler_type="predictive")
    return _run_and_save(workload_mv, scheduler, collector, "predictive", pattern, seed, workload_run)


def run_comparison(pattern: str, steps: int, seed: int) -> dict:
    """Run both schedulers on the same workload and return both results."""
    r_run = run_reactive(pattern=pattern, steps=steps, seed=seed)
    p_run = run_predictive(pattern=pattern, steps=steps, seed=seed)
    return {"reactive_id": r_run.pk, "predictive_id": p_run.pk}
