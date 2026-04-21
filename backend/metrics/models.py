"""metrics/models.py"""

from django.db import models
from scheduler.models import SchedulerRun


class MetricRecord(models.Model):
    """Aggregate per-run metric snapshot (used for dashboard KPIs)."""
    scheduler_run    = models.OneToOneField(SchedulerRun, on_delete=models.CASCADE,
                                            related_name="metric_record", null=True, blank=True)
    scheduler_type   = models.CharField(max_length=20)
    pattern          = models.CharField(max_length=20)
    steps            = models.IntegerField()
    overload_events  = models.IntegerField()
    overload_rate    = models.FloatField()
    avg_cpu          = models.FloatField()
    avg_capacity     = models.FloatField()
    total_cost       = models.FloatField()
    scale_up_count   = models.IntegerField(default=0)
    scale_down_count = models.IntegerField(default=0)
    recorded_at      = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-recorded_at"]

    def __str__(self):
        return f"{self.scheduler_type} | {self.pattern} | overloads={self.overload_events}"
