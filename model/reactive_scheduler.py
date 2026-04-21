"""
model/reactive_scheduler.py
----------------------------
Threshold-based reactive scheduler — scales AFTER overload is observed.
Mirrors real-world reactive autoscalers (AWS/GCP default behaviour).
"""

from model.metrics_collector import CAPACITY_PER_UNIT


class ReactiveScheduler:
    """
    Parameters
    ----------
    scale_up_threshold   : CPU % above which we scale up
    scale_down_threshold : CPU % below which we scale down
    min_capacity         : minimum resource units
    max_capacity         : maximum resource units
    cooldown_steps       : steps to wait between scaling actions
    """

    def __init__(
        self,
        scale_up_threshold: float = 70.0,
        scale_down_threshold: float = 30.0,
        min_capacity: int = 1,
        max_capacity: int = 20,
        cooldown_steps: int = 5,
    ):
        self.scale_up_threshold   = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_capacity         = min_capacity
        self.max_capacity         = max_capacity
        self.cooldown_steps       = cooldown_steps
        self.capacity             = min_capacity
        self._cooldown_counter    = 0

    def decide(self, current_load: float) -> int:
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return self.capacity

        effective_cpu = (current_load / max(self.capacity * CAPACITY_PER_UNIT, 1)) * 100.0

        if effective_cpu > self.scale_up_threshold:
            self.capacity = min(self.capacity + 1, self.max_capacity)
            self._cooldown_counter = self.cooldown_steps
        elif effective_cpu < self.scale_down_threshold:
            self.capacity = max(self.capacity - 1, self.min_capacity)
            self._cooldown_counter = self.cooldown_steps

        return self.capacity

    def reset(self):
        self.capacity          = self.min_capacity
        self._cooldown_counter = 0
