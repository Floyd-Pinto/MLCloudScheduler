"""metrics/views.py"""

from rest_framework.views import APIView
from rest_framework.response import Response

from scheduler.models import SchedulerRun
from scheduler.serializers import SchedulerRunSummarySerializer


class MetricListView(APIView):
    """GET /api/metrics/ — all scheduler run summaries (used as metrics feed)."""

    def get(self, request):
        scheduler_type = request.query_params.get("type")
        pattern        = request.query_params.get("pattern")
        qs = SchedulerRun.objects.all()
        if scheduler_type:
            qs = qs.filter(scheduler_type=scheduler_type)
        if pattern:
            qs = qs.filter(pattern=pattern)
        qs = qs[:200]
        return Response(SchedulerRunSummarySerializer(qs, many=True).data)


class MetricSummaryView(APIView):
    """GET /api/metrics/summary/ — aggregate KPIs across all runs."""

    def get(self, request):
        from django.db.models import Avg, Sum, Count, Min

        stats = SchedulerRun.objects.aggregate(
            total_runs    = Count("id"),
            avg_overload  = Avg("overload_rate"),
            avg_cpu       = Avg("avg_cpu"),
            total_cost    = Sum("total_cost"),
        )

        reactive_stats = SchedulerRun.objects.filter(scheduler_type="reactive").aggregate(
            avg_overload = Avg("overload_rate"),
            avg_cpu      = Avg("avg_cpu"),
            total_runs   = Count("id"),
        )
        predictive_stats = SchedulerRun.objects.filter(scheduler_type="predictive").aggregate(
            avg_overload = Avg("overload_rate"),
            avg_cpu      = Avg("avg_cpu"),
            total_runs   = Count("id"),
        )

        return Response({
            "overall":    stats,
            "reactive":   reactive_stats,
            "predictive": predictive_stats,
        })
