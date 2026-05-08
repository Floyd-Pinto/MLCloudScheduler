"""
model/metrics_collector.py
--------------------------
Records per-step simulation metrics across three resource dimensions
and computes aggregate statistics for comparing reactive vs. predictive
schedulers.

Each step records:
  cpu_usage, memory_usage, network_io, capacity,
  overload_cpu, overload_memory, overload_network, overload_any,
  scaling_action, trigger_resource, cost
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


CAPACITY_PER_UNIT: float = 10.0   # each resource unit handles 10 workload units at 100%


@dataclass
class StepRecord:
    """Per-step record for multi-resource scheduler simulation."""
    time_step:        int
    workload:         float
    cpu_usage:        float      # percent (capped at 100)
    memory_usage:     float      # percent (capped at 100)
    network_io:       float      # percent (capped at 100)
    capacity:         int
    overload_cpu:     bool
    overload_memory:  bool
    overload_network: bool
    overload_any:     bool
    action:           str        # "scale_up" | "scale_down" | "hold"
    trigger_resource: str        # which resource triggered the action
    scheduler_type:   str        # "reactive" | "predictive"
    cost:             float      # cost for this step


class MetricsCollector:
    """Collects per-step multi-resource records and computes summary statistics."""

    def __init__(self, overload_threshold_cpu: float = 80.0,
                 overload_threshold_memory: float = 80.0,
                 overload_threshold_network: float = 85.0,
                 cost_per_unit: float = 1.0,
                 scheduler_type: str = "reactive"):
        self.overload_threshold_cpu     = overload_threshold_cpu
        self.overload_threshold_memory  = overload_threshold_memory
        self.overload_threshold_network = overload_threshold_network
        self.cost_per_unit = cost_per_unit
        self.scheduler_type = scheduler_type
        self._records: list[StepRecord] = []
        self._prev_capacity: int | None = None

    def record(self, time_step: int, workload: float, capacity: int,
               cpu_usage: float = 0.0, memory_usage: float = 0.0,
               network_io: float = 0.0, trigger_resource: str = "none"):
        """
        Record a single time step with multi-resource metrics.

        Parameters
        ----------
        workload : float
            Raw workload intensity for the step.
        cpu_usage, memory_usage, network_io : float
            Current resource utilisation percentages (0–100).
        trigger_resource : str
            Which resource triggered a scaling action (if any).
        """
        overload_cpu     = cpu_usage > self.overload_threshold_cpu
        overload_memory  = memory_usage > self.overload_threshold_memory
        overload_network = network_io > self.overload_threshold_network
        overload_any     = overload_cpu or overload_memory or overload_network

        if self._prev_capacity is None:
            action = "hold"
        elif capacity > self._prev_capacity:
            action = "scale_up"
        elif capacity < self._prev_capacity:
            action = "scale_down"
        else:
            action = "hold"
        self._prev_capacity = capacity

        step_cost = float(capacity) * self.cost_per_unit

        self._records.append(
            StepRecord(
                time_step=time_step,
                workload=round(workload, 4),
                cpu_usage=round(cpu_usage, 4),
                memory_usage=round(memory_usage, 4),
                network_io=round(network_io, 4),
                capacity=capacity,
                overload_cpu=overload_cpu,
                overload_memory=overload_memory,
                overload_network=overload_network,
                overload_any=overload_any,
                action=action,
                trigger_resource=trigger_resource,
                scheduler_type=self.scheduler_type,
                cost=step_cost,
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self._records])

    def summary(self) -> dict:
        df = self.to_dataframe()
        if len(df) == 0:
            return {"scheduler_type": self.scheduler_type, "total_steps": 0}
        return {
            "scheduler_type":    self.scheduler_type,
            "total_steps":       len(df),
            "overload_cpu":      int(df["overload_cpu"].sum()),
            "overload_memory":   int(df["overload_memory"].sum()),
            "overload_network":  int(df["overload_network"].sum()),
            "overload_any":      int(df["overload_any"].sum()),
            "overload_events":   int(df["overload_any"].sum()),  # backward compat
            "overload_rate":     round(float(df["overload_any"].mean()) * 100, 2),
            "avg_cpu":           round(float(df["cpu_usage"].mean()), 2),
            "avg_memory":        round(float(df["memory_usage"].mean()), 2),
            "avg_network":       round(float(df["network_io"].mean()), 2),
            "avg_capacity":      round(float(df["capacity"].mean()), 2),
            "total_cost":        round(float(df["cost"].sum()), 2),
            "scale_up_count":    int((df["action"] == "scale_up").sum()),
            "scale_down_count":  int((df["action"] == "scale_down").sum()),
        }

    def reset(self):
        self._records.clear()
        self._prev_capacity = None
