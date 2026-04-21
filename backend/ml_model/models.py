"""backend/ml_model/models.py — Updated with ModelComparisonResult."""
from django.db import models


class ModelTrainingRun(models.Model):
    MODEL_TYPES = [
        ("gbr",      "GradientBoosting"),
        ("lstm",     "LSTM (PyTorch)"),
        ("arima",    "ARIMA"),
        ("combined", "Combined LSTM+ARIMA"),
    ]
    STATUS_CHOICES = [
        ("running",   "Running"),
        ("completed", "Completed"),
        ("failed",    "Failed"),
    ]
    model_type  = models.CharField(max_length=32, choices=MODEL_TYPES)
    status      = models.CharField(max_length=16, choices=STATUS_CHOICES, default="running")
    rmse        = models.FloatField(null=True, blank=True)
    mae         = models.FloatField(null=True, blank=True)
    r2          = models.FloatField(null=True, blank=True)
    extra_info  = models.JSONField(default=dict, blank=True)   # stores w_lstm/w_arima for combined
    error_msg   = models.TextField(blank=True, default="")
    started_at  = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        return f"{self.model_type} — {self.status} ({self.started_at:%Y-%m-%d %H:%M})"


class ModelComparisonResult(models.Model):
    """Stores a side-by-side evaluation of all 4 models on the same test series."""
    created_at    = models.DateTimeField(auto_now_add=True)
    series_length = models.IntegerField(default=0)
    pattern       = models.CharField(max_length=32, default="combined")
    seed          = models.IntegerField(default=42)

    # GBR
    gbr_rmse  = models.FloatField(null=True, blank=True)
    gbr_mae   = models.FloatField(null=True, blank=True)
    gbr_r2    = models.FloatField(null=True, blank=True)
    # LSTM
    lstm_rmse = models.FloatField(null=True, blank=True)
    lstm_mae  = models.FloatField(null=True, blank=True)
    lstm_r2   = models.FloatField(null=True, blank=True)
    # ARIMA
    arima_rmse = models.FloatField(null=True, blank=True)
    arima_mae  = models.FloatField(null=True, blank=True)
    arima_r2   = models.FloatField(null=True, blank=True)
    # Combined
    combined_rmse = models.FloatField(null=True, blank=True)
    combined_mae  = models.FloatField(null=True, blank=True)
    combined_r2   = models.FloatField(null=True, blank=True)

    best_model = models.CharField(max_length=32, default="")

    class Meta:
        ordering = ["-created_at"]
