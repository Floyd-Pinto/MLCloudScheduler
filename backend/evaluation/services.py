"""evaluation/services.py"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scheduler.services import run_reactive, run_predictive
from scheduler.models import SchedulerRun
from .models import EvaluationResult


def run_full_evaluation(pattern: str, steps: int, seed: int) -> EvaluationResult:
    r_run = run_reactive(pattern=pattern, steps=steps, seed=seed)
    p_run = run_predictive(pattern=pattern, steps=steps, seed=seed)

    overload_reduction = 0.0
    if r_run.overload_events > 0:
        overload_reduction = (
            (r_run.overload_events - p_run.overload_events) / r_run.overload_events
        ) * 100.0

    cost_diff = p_run.total_cost - r_run.total_cost

    result = EvaluationResult.objects.create(
        pattern            = pattern,
        steps              = steps,
        seed               = seed,
        r_overload_events  = r_run.overload_events,
        r_overload_rate    = r_run.overload_rate,
        r_avg_cpu          = r_run.avg_cpu,
        r_total_cost       = r_run.total_cost,
        r_scale_up         = r_run.scale_up_count,
        r_scale_down       = r_run.scale_down_count,
        p_overload_events  = p_run.overload_events,
        p_overload_rate    = p_run.overload_rate,
        p_avg_cpu          = p_run.avg_cpu,
        p_total_cost       = p_run.total_cost,
        p_scale_up         = p_run.scale_up_count,
        p_scale_down       = p_run.scale_down_count,
        overload_reduction = round(overload_reduction, 2),
        cost_difference    = round(cost_diff, 2),
        reactive_run_id    = r_run.pk,
        predictive_run_id  = p_run.pk,
    )
    return result


def get_latest_comparison(pattern: str = None) -> dict:
    qs = EvaluationResult.objects.all()
    if pattern:
        qs = qs.filter(pattern=pattern)
    result = qs.first()
    if not result:
        return {}
    return {
        "id":                 result.pk,
        "pattern":            result.pattern,
        "steps":              result.steps,
        "overload_reduction": result.overload_reduction,
        "cost_difference":    result.cost_difference,
        "reactive": {
            "overload_events":  result.r_overload_events,
            "overload_rate":    result.r_overload_rate,
            "avg_cpu":          result.r_avg_cpu,
            "total_cost":       result.r_total_cost,
            "scale_up":         result.r_scale_up,
            "scale_down":       result.r_scale_down,
        },
        "predictive": {
            "overload_events":  result.p_overload_events,
            "overload_rate":    result.p_overload_rate,
            "avg_cpu":          result.p_avg_cpu,
            "total_cost":       result.p_total_cost,
            "scale_up":         result.p_scale_up,
            "scale_down":       result.p_scale_down,
        },
    }
