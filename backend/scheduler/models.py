"""scheduler/models.py"""

from django.db import models
from simulation.models import WorkloadRun


class SchedulerRun(models.Model):
    TYPE_CHOICES = [("reactive", "Reactive"), ("predictive", "Predictive")]

    workload_run    = models.ForeignKey(WorkloadRun, on_delete=models.CASCADE,
                                        related_name="scheduler_runs", null=True, blank=True)
    scheduler_type  = models.CharField(max_length=20, choices=TYPE_CHOICES)
    pattern         = models.CharField(max_length=20, default="combined")
    steps           = models.IntegerField(default=200)
    seed            = models.IntegerField(default=42)

    # Summary stats (filled in after run completes)
    overload_events  = models.IntegerField(default=0)
    overload_rate    = models.FloatField(default=0.0)
    avg_cpu          = models.FloatField(default=0.0)
    avg_capacity     = models.FloatField(default=0.0)
    total_cost       = models.FloatField(default=0.0)
    scale_up_count   = models.IntegerField(default=0)
    scale_down_count = models.IntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.scheduler_type} | {self.pattern} | overloads={self.overload_events}"


class SchedulerAction(models.Model):
    ACTION_CHOICES = [("scale_up", "Scale Up"), ("scale_down", "Scale Down"), ("hold", "Hold")]

    run       = models.ForeignKey(SchedulerRun, on_delete=models.CASCADE, related_name="actions")
    time_step = models.IntegerField()
    workload  = models.FloatField()
    capacity  = models.IntegerField()
    cpu_usage = models.FloatField()
    overloaded= models.BooleanField(default=False)
    action    = models.CharField(max_length=15, choices=ACTION_CHOICES, default="hold")

    class Meta:
        ordering = ["time_step"]
