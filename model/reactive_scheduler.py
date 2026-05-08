"""
model/reactive_scheduler.py
----------------------------
Multi-resource threshold-based reactive scheduler — scales AFTER
overload is observed on any of the three monitored resources.

Scale-out trigger:
  CPU > 80% OR memory > 80% OR network_io > 85% for 2 consecutive steps

Scale-in trigger:
  ALL three resources < 40% for 5 consecutive steps

Mirrors real-world reactive autoscalers (AWS/GCP default behaviour)
extended to multi-dimensional resource monitoring.
"""

from model.metrics_collector import CAPACITY_PER_UNIT


class ReactiveScheduler:
    """
    Multi-resource reactive scheduler with per-resource overload detection.

    Parameters
    ----------
    cpu_threshold    : CPU % above which overload is flagged
    mem_threshold    : Memory % above which overload is flagged
    net_threshold    : Network I/O % above which overload is flagged
    scale_down_threshold : All resources below this % triggers scale-in
    min_capacity     : minimum resource units
    max_capacity     : maximum resource units
    consec_up        : consecutive overload steps before scaling up
    consec_down      : consecutive low-util steps before scaling down
    """

    def __init__(
        self,
        cpu_threshold: float = 80.0,
        mem_threshold: float = 80.0,
        net_threshold: float = 85.0,
        scale_down_threshold: float = 40.0,
        min_capacity: int = 1,
        max_capacity: int = 20,
        consec_up: int = 2,
        consec_down: int = 5,
        # Legacy parameter for backward compatibility
        scale_up_threshold: float | None = None,
        cooldown_steps: int = 5,
        scale_down_threshold_legacy: float | None = None,
    ):
        self.cpu_threshold   = cpu_threshold
        self.mem_threshold   = mem_threshold
        self.net_threshold   = net_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_capacity    = min_capacity
        self.max_capacity    = max_capacity
        self.consec_up       = consec_up
        self.consec_down     = consec_down
        self.capacity        = min_capacity

        self._overload_counter = 0   # consecutive overload steps
        self._low_counter      = 0   # consecutive low-utilisation steps
        self._last_trigger     = "none"

    def decide(self, current_load: float,
               cpu_pct: float | None = None,
               mem_pct: float | None = None,
               net_pct: float | None = None) -> int:
        """
        Make a scaling decision based on current resource utilisation.

        Parameters
        ----------
        current_load : float
            Raw workload value (used to derive CPU if cpu_pct not provided).
        cpu_pct, mem_pct, net_pct : float, optional
            Explicit per-resource utilisation percentages.

        Returns
        -------
        int
            Updated capacity (resource units).
        """
        # Derive CPU from workload if not explicitly provided
        if cpu_pct is None:
            cpu_pct = min((current_load / max(self.capacity * CAPACITY_PER_UNIT, 1)) * 100.0, 100.0)
        if mem_pct is None:
            mem_pct = cpu_pct * 0.7
        if net_pct is None:
            net_pct = cpu_pct * 0.5

        # Check per-resource overloads
        overload_cpu = cpu_pct > self.cpu_threshold
        overload_mem = mem_pct > self.mem_threshold
        overload_net = net_pct > self.net_threshold
        any_overload = overload_cpu or overload_mem or overload_net

        # Identify trigger resource
        triggers = []
        if overload_cpu: triggers.append("cpu")
        if overload_mem: triggers.append("memory")
        if overload_net: triggers.append("network")
        self._last_trigger = ",".join(triggers) if triggers else "none"

        # Consecutive overload tracking for scale-up
        if any_overload:
            self._overload_counter += 1
            self._low_counter = 0
        else:
            self._overload_counter = 0

        # Consecutive low-utilisation tracking for scale-down
        all_low = (cpu_pct < self.scale_down_threshold and
                   mem_pct < self.scale_down_threshold and
                   net_pct < self.scale_down_threshold)
        if all_low:
            self._low_counter += 1
        else:
            self._low_counter = 0

        # Scale-out: overload for consec_up consecutive steps
        if self._overload_counter >= self.consec_up:
            self.capacity = min(self.capacity + 1, self.max_capacity)
            self._overload_counter = 0

        # Scale-in: all resources low for consec_down consecutive steps
        elif self._low_counter >= self.consec_down:
            self.capacity = max(self.capacity - 1, self.min_capacity)
            self._low_counter = 0
            self._last_trigger = "scale_in"

        return self.capacity

    @property
    def last_trigger(self) -> str:
        return self._last_trigger

    def reset(self):
        self.capacity          = self.min_capacity
        self._overload_counter = 0
        self._low_counter      = 0
        self._last_trigger     = "none"
