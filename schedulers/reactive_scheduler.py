"""
schedulers/reactive_scheduler.py
---------------------------------
Baseline scheduler: scales resources only AFTER CPU/workload
crosses a static threshold — mirrors real-world reactive autoscalers.
"""


class ReactiveScheduler:
    """
    Threshold-based reactive scheduler.

    Parameters
    ----------
    scale_up_threshold   : float  – CPU % above which we scale up
    scale_down_threshold : float  – CPU % below which we scale down
    min_capacity         : int    – minimum number of resource units
    max_capacity         : int    – maximum number of resource units
    cooldown_steps       : int    – steps to wait between scaling actions
    """

    def __init__(
        self,
        scale_up_threshold: float = 70.0,
        scale_down_threshold: float = 30.0,
        min_capacity: int = 1,
        max_capacity: int = 20,
        cooldown_steps: int = 5,
    ):
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.cooldown_steps = cooldown_steps

        self.capacity = min_capacity
        self._cooldown_counter = 0

    # ------------------------------------------------------------------
    # Must match MetricsCollector.CAPACITY_PER_UNIT
    CAPACITY_PER_UNIT: float = 10.0

    def decide(self, current_load: float) -> int:
        """
        Given the *current* observed load, return the new capacity.

        Scaling is delayed by cooldown_steps after each action to avoid
        thrashing — same behaviour as AWS/GCP autoscalers.
        """
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return self.capacity

        # Convert raw load into effective CPU% for threshold comparison
        effective_cpu = (current_load / max(self.capacity * self.CAPACITY_PER_UNIT, 1)) * 100.0

        if effective_cpu > self.scale_up_threshold:
            self.capacity = min(self.capacity + 1, self.max_capacity)
            self._cooldown_counter = self.cooldown_steps
        elif effective_cpu < self.scale_down_threshold:
            self.capacity = max(self.capacity - 1, self.min_capacity)
            self._cooldown_counter = self.cooldown_steps

        return self.capacity

    def reset(self):
        self.capacity = self.min_capacity
        self._cooldown_counter = 0
