"""
metrics/collector.py
--------------------
Records per-step simulation metrics and computes aggregate statistics
for comparing reactive vs. predictive schedulers.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class StepRecord:
    time_step:   int
    workload:    float
    capacity:    int
    cpu_usage:   float   # workload / capacity  (capped at 100 %)
    overloaded:  bool    # True when cpu_usage > overload_threshold


class MetricsCollector:
    """
    Collects per-step records and computes summary statistics.

    Parameters
    ----------
    overload_threshold : float – CPU % considered an overload event
    cost_per_unit      : float – simulated cost per resource unit per step
    """

    def __init__(self, overload_threshold: float = 80.0,
                 cost_per_unit: float = 1.0):
        self.overload_threshold = overload_threshold
        self.cost_per_unit = cost_per_unit
        self._records: list[StepRecord] = []

    # ------------------------------------------------------------------
    # Each resource unit handles this many workload units at 100% utilization
    CAPACITY_PER_UNIT: float = 10.0

    def record(self, time_step: int, workload: float, capacity: int):
        cpu = min((workload / max(capacity * self.CAPACITY_PER_UNIT, 1)) * 100.0, 100.0)
        overloaded = cpu > self.overload_threshold
        self._records.append(
            StepRecord(time_step, workload, capacity, cpu, overloaded)
        )

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self._records])

    def summary(self) -> dict:
        df = self.to_dataframe()
        return {
            "total_steps":       len(df),
            "overload_events":   int(df["overloaded"].sum()),
            "overload_rate_%":   round(df["overloaded"].mean() * 100, 2),
            "avg_cpu_%":         round(df["cpu_usage"].mean(), 2),
            "avg_capacity":      round(df["capacity"].mean(), 2),
            "total_cost":        round(df["capacity"].sum() * self.cost_per_unit, 2),
        }

    def reset(self):
        self._records.clear()
