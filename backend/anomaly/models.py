"""anomaly/models.py — Anomaly detection log entries."""

from django.db import models


class AnomalyLogEntry(models.Model):
    """Stores anomaly detection results from scheduler runs."""
    created_at      = models.DateTimeField(auto_now_add=True)
    time_step       = models.IntegerField()
    cpu_usage       = models.FloatField()
    memory_usage    = models.FloatField()
    network_io      = models.FloatField()
    is_anomaly      = models.BooleanField(default=False)
    z_score_flag    = models.BooleanField(default=False)
    iforest_flag    = models.BooleanField(default=False)
    pattern         = models.CharField(max_length=32, default="combined")
    scheduler_type  = models.CharField(max_length=20, default="predictive")

    class Meta:
        ordering = ["-created_at", "time_step"]

    def __str__(self):
        return f"t={self.time_step} anomaly={self.is_anomaly}"
