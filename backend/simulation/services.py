"""simulation/services.py — Updated for Phase 2 multi-resource generation."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from model.workload_generator import generate, generate_multivariate, generate_from_real_data

from .models import WorkloadRun, WorkloadDataPoint


def generate_and_save(pattern: str, steps: int, seed: int, label: str = "") -> WorkloadRun:
    """Generate multi-resource workload and save to DB."""
    # Route real-data patterns to the appropriate loader
    if pattern == "google_trace":
        workload_mv = generate_from_real_data(source="google", steps=steps, start_offset=seed)
    elif pattern == "alibaba_trace":
        workload_mv = generate_from_real_data(source="alibaba", steps=steps, start_offset=seed)
    else:
        workload_mv = generate_multivariate(pattern=pattern, steps=steps, seed=seed)

    run = WorkloadRun.objects.create(pattern=pattern, steps=steps, seed=seed, label=label)

    datapoints = [
        WorkloadDataPoint(
            run=run,
            time_step=int(t),
            workload=float(workload_mv[t, 0]),
            memory_usage=float(workload_mv[t, 1]),
            network_io=float(workload_mv[t, 2]),
        )
        for t in range(len(workload_mv))
    ]
    WorkloadDataPoint.objects.bulk_create(datapoints)
    return run

