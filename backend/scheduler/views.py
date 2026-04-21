"""scheduler/views.py"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import SchedulerRun
from .serializers import (RunSchedulerSerializer, SchedulerRunSummarySerializer,
                           SchedulerRunDetailSerializer)
from .services import run_reactive, run_predictive, run_comparison

logger = logging.getLogger(__name__)


class RunReactiveView(APIView):
    def post(self, request):
        ser = RunSchedulerSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        d = ser.validated_data
        try:
            run = run_reactive(d["pattern"], d["steps"], d["seed"])
            return Response(SchedulerRunDetailSerializer(run).data, status=status.HTTP_201_CREATED)
        except Exception as exc:
            logger.exception("Reactive scheduler failed")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RunPredictiveView(APIView):
    def post(self, request):
        ser = RunSchedulerSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        d = ser.validated_data
        try:
            run = run_predictive(d["pattern"], d["steps"], d["seed"])
            return Response(SchedulerRunDetailSerializer(run).data, status=status.HTTP_201_CREATED)
        except Exception as exc:
            logger.exception("Predictive scheduler failed")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RunComparisonView(APIView):
    def post(self, request):
        ser = RunSchedulerSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        d = ser.validated_data
        try:
            ids = run_comparison(d["pattern"], d["steps"], d["seed"])
            r_run = SchedulerRun.objects.get(pk=ids["reactive_id"])
            p_run = SchedulerRun.objects.get(pk=ids["predictive_id"])
            return Response({
                "reactive":   SchedulerRunDetailSerializer(r_run).data,
                "predictive": SchedulerRunDetailSerializer(p_run).data,
            }, status=status.HTTP_201_CREATED)
        except Exception as exc:
            logger.exception("Comparison run failed")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SchedulerRunListView(APIView):
    def get(self, request):
        runs = SchedulerRun.objects.all()[:200]
        return Response(SchedulerRunSummarySerializer(runs, many=True).data)


class SchedulerRunDetailView(APIView):
    def get(self, request, pk):
        try:
            run = SchedulerRun.objects.get(pk=pk)
        except SchedulerRun.DoesNotExist:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(SchedulerRunDetailSerializer(run).data)
