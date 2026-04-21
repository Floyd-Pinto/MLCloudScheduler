"""simulation/models.py"""

from django.db import models


class WorkloadRun(models.Model):
    PATTERN_CHOICES = [
        ("gradual",  "Gradual"),
        ("spike",    "Spike"),
        ("periodic", "Periodic"),
        ("combined", "Combined"),
    ]
    pattern    = models.CharField(max_length=20, choices=PATTERN_CHOICES, default="combined")
    steps      = models.IntegerField(default=200)
    seed       = models.IntegerField(default=42)
    created_at = models.DateTimeField(auto_now_add=True)
    label      = models.CharField(max_length=120, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"WorkloadRun({self.pattern}, steps={self.steps}, id={self.pk})"


class WorkloadDataPoint(models.Model):
    run       = models.ForeignKey(WorkloadRun, on_delete=models.CASCADE, related_name="datapoints")
    time_step = models.IntegerField()
    workload  = models.FloatField()

    class Meta:
        ordering = ["time_step"]

    def __str__(self):
        return f"t={self.time_step} w={self.workload:.2f}"
