"""simulation/services.py — workload generation business logic."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from model.workload_generator import generate

from .models import WorkloadRun, WorkloadDataPoint


def generate_and_save(pattern: str, steps: int, seed: int, label: str = "") -> WorkloadRun:
    workload: np.ndarray = generate(pattern=pattern, steps=steps, seed=seed)

    run = WorkloadRun.objects.create(pattern=pattern, steps=steps, seed=seed, label=label)

    datapoints = [
        WorkloadDataPoint(run=run, time_step=int(t), workload=float(w))
        for t, w in enumerate(workload)
    ]
    WorkloadDataPoint.objects.bulk_create(datapoints)
    return run
