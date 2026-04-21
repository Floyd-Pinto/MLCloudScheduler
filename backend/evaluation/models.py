"""evaluation/models.py"""

from django.db import models


class EvaluationResult(models.Model):
    pattern             = models.CharField(max_length=20)
    steps               = models.IntegerField()
    seed                = models.IntegerField()

    # Reactive
    r_overload_events   = models.IntegerField()
    r_overload_rate     = models.FloatField()
    r_avg_cpu           = models.FloatField()
    r_total_cost        = models.FloatField()
    r_scale_up          = models.IntegerField(default=0)
    r_scale_down        = models.IntegerField(default=0)

    # Predictive
    p_overload_events   = models.IntegerField()
    p_overload_rate     = models.FloatField()
    p_avg_cpu           = models.FloatField()
    p_total_cost        = models.FloatField()
    p_scale_up          = models.IntegerField(default=0)
    p_scale_down        = models.IntegerField(default=0)

    # Derived
    overload_reduction  = models.FloatField()   # % improvement
    cost_difference     = models.FloatField()   # predictive - reactive

    reactive_run_id     = models.IntegerField(null=True, blank=True)
    predictive_run_id   = models.IntegerField(null=True, blank=True)
    created_at          = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Eval [{self.pattern}] overload_reduction={self.overload_reduction:.1f}%"
