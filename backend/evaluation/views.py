"""evaluation/views.py"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import EvaluationResult
from .serializers import EvaluationResultSerializer, RunEvaluationSerializer
from .services import run_full_evaluation, get_latest_comparison

logger = logging.getLogger(__name__)


class RunEvaluationView(APIView):
    """POST /api/evaluation/run/ — run both schedulers and save comparison."""

    def post(self, request):
        ser = RunEvaluationSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        d = ser.validated_data
        try:
            result = run_full_evaluation(d["pattern"], d["steps"], d["seed"])
            return Response(EvaluationResultSerializer(result).data, status=status.HTTP_201_CREATED)
        except Exception as exc:
            logger.exception("Evaluation failed")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class EvaluationListView(APIView):
    """GET /api/evaluation/ — list all evaluation results."""

    def get(self, request):
        results = EvaluationResult.objects.all()[:50]
        return Response(EvaluationResultSerializer(results, many=True).data)


class EvaluationComparisonView(APIView):
    """GET /api/evaluation/comparison/?pattern=combined — latest comparison."""

    def get(self, request):
        pattern = request.query_params.get("pattern")
        data    = get_latest_comparison(pattern=pattern)
        if not data:
            return Response({"message": "No evaluation results found. Run /api/evaluation/run/ first."},
                            status=status.HTTP_404_NOT_FOUND)
        return Response(data)
