"""simulation/views.py"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import WorkloadRun
from .serializers import (WorkloadRunSerializer, WorkloadRunListSerializer,
                           GenerateWorkloadSerializer)
from .services import generate_and_save

logger = logging.getLogger(__name__)


class GenerateWorkloadView(APIView):
    """POST /api/simulation/generate/ — generate synthetic workload."""

    def post(self, request):
        ser = GenerateWorkloadSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        d = ser.validated_data
        try:
            run = generate_and_save(
                pattern=d["pattern"],
                steps=d["steps"],
                seed=d["seed"],
                label=d.get("label", ""),
            )
            return Response(WorkloadRunSerializer(run).data, status=status.HTTP_201_CREATED)
        except Exception as exc:
            logger.exception("Workload generation failed")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class WorkloadRunListView(APIView):
    """GET /api/simulation/runs/ — list all workload runs."""

    def get(self, request):
        runs = WorkloadRun.objects.all()[:100]
        return Response(WorkloadRunListSerializer(runs, many=True).data)


class WorkloadRunDetailView(APIView):
    """GET /api/simulation/runs/{pk}/ — full run with datapoints."""

    def get(self, request, pk):
        try:
            run = WorkloadRun.objects.get(pk=pk)
        except WorkloadRun.DoesNotExist:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(WorkloadRunSerializer(run).data)

    def delete(self, request, pk):
        try:
            run = WorkloadRun.objects.get(pk=pk)
            run.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except WorkloadRun.DoesNotExist:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
