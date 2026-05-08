"""anomaly/views.py"""

from rest_framework.views import APIView
from rest_framework.response import Response

from .models import AnomalyLogEntry
from .serializers import AnomalyLogEntrySerializer


class AnomalyLogListView(APIView):
    """GET /api/anomaly/logs/ — list anomaly detection logs."""

    def get(self, request):
        qs = AnomalyLogEntry.objects.all()[:500]
        return Response(AnomalyLogEntrySerializer(qs, many=True).data)


class AnomalyLogSummaryView(APIView):
    """GET /api/anomaly/summary/ — aggregate anomaly statistics."""

    def get(self, request):
        total = AnomalyLogEntry.objects.count()
        anomalies = AnomalyLogEntry.objects.filter(is_anomaly=True).count()
        return Response({
            "total_checked": total,
            "total_anomalies": anomalies,
            "anomaly_rate": round(anomalies / max(total, 1) * 100, 2),
        })
