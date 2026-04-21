"""
model/metrics_collector.py
--------------------------
Records per-step simulation metrics and computes aggregate statistics
for comparing reactive vs. predictive schedulers.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


CAPACITY_PER_UNIT: float = 10.0   # each resource unit handles 10 workload units at 100%


@dataclass
class StepRecord:
    time_step:      int
    workload:       float
    capacity:       int
    cpu_usage:      float   # percent  (capped at 100)
    overloaded:     bool
    action:         str     # "scale_up" | "scale_down" | "hold"
    scheduler_type: str     # "reactive" | "predictive"


class MetricsCollector:
    """Collects per-step records and computes summary statistics."""

    def __init__(self, overload_threshold: float = 80.0,
                 cost_per_unit: float = 1.0,
                 scheduler_type: str = "reactive"):
        self.overload_threshold = overload_threshold
        self.cost_per_unit = cost_per_unit
        self.scheduler_type = scheduler_type
        self._records: list[StepRecord] = []
        self._prev_capacity: int | None = None

    def record(self, time_step: int, workload: float, capacity: int):
        cpu = min((workload / max(capacity * CAPACITY_PER_UNIT, 1)) * 100.0, 100.0)
        overloaded = cpu > self.overload_threshold
        if self._prev_capacity is None:
            action = "hold"
        elif capacity > self._prev_capacity:
            action = "scale_up"
        elif capacity < self._prev_capacity:
            action = "scale_down"
        else:
            action = "hold"
        self._prev_capacity = capacity
        self._records.append(
            StepRecord(time_step, round(workload, 4), capacity,
                       round(cpu, 4), overloaded, action, self.scheduler_type)
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self._records])

    def summary(self) -> dict:
        df = self.to_dataframe()
        return {
            "scheduler_type":  self.scheduler_type,
            "total_steps":     len(df),
            "overload_events": int(df["overloaded"].sum()),
            "overload_rate":   round(float(df["overloaded"].mean()) * 100, 2),
            "avg_cpu":         round(float(df["cpu_usage"].mean()), 2),
            "avg_capacity":    round(float(df["capacity"].mean()), 2),
            "total_cost":      round(float(df["capacity"].sum()) * self.cost_per_unit, 2),
            "scale_up_count":  int((df["action"] == "scale_up").sum()),
            "scale_down_count":int((df["action"] == "scale_down").sum()),
        }

    def reset(self):
        self._records.clear()
        self._prev_capacity = None
