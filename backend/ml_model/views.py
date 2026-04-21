"""backend/ml_model/views.py — all ML API views."""
from rest_framework.views   import APIView
from rest_framework.response import Response
from rest_framework          import status

from .models       import ModelTrainingRun, ModelComparisonResult
from .serializers  import (ModelTrainingRunSerializer, ModelComparisonResultSerializer,
                            TrainRequestSerializer, PredictRequestSerializer)
from .services     import (trigger_training, trigger_train_all,
                            run_inference, run_inference_all,
                            get_model_status, compare_all_models)


class TrainView(APIView):
    """POST /api/ml/train/  — train one or all models."""
    def post(self, request):
        ser = TrainRequestSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)

        mt = ser.validated_data["model_type"]

        if mt == "all":
            records = trigger_train_all()
            return Response(
                ModelTrainingRunSerializer(records, many=True).data,
                status=status.HTTP_201_CREATED,
            )

        record = trigger_training(mt)
        return Response(ModelTrainingRunSerializer(record).data, status=status.HTTP_201_CREATED)


class ModelStatusView(APIView):
    """GET /api/ml/status/ — readiness + latest metrics for all models."""
    def get(self, request):
        return Response(get_model_status())


class ModelHistoryView(APIView):
    """GET /api/ml/history/ — all training run records."""
    def get(self, request):
        mt = request.query_params.get("model_type")
        qs = ModelTrainingRun.objects.all()
        if mt:
            qs = qs.filter(model_type=mt)
        return Response(ModelTrainingRunSerializer(qs, many=True).data)


class PredictView(APIView):
    """POST /api/ml/predict/ — single-model prediction."""
    def post(self, request):
        ser = PredictRequestSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        result = run_inference(**ser.validated_data)
        if "error" in result:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        return Response(result)


class PredictAllView(APIView):
    """POST /api/ml/predict-all/ — all models in one call."""
    def post(self, request):
        history = request.data.get("history")
        if not history or len(history) < 10:
            return Response({"error": "Provide at least 10 history values."}, status=400)
        return Response(run_inference_all(history))


class ModelCompareView(APIView):
    """
    POST /api/ml/compare-models/ — evaluate all models on a workload, return metrics + chart data.
    GET  /api/ml/compare-models/ — list saved comparison results.
    """
    def get(self, request):
        qs = ModelComparisonResult.objects.all()
        return Response(ModelComparisonResultSerializer(qs, many=True).data)

    def post(self, request):
        pattern = request.data.get("pattern", "combined")
        steps   = int(request.data.get("steps", 300))
        seed    = int(request.data.get("seed",    42))
        result  = compare_all_models(pattern=pattern, steps=steps, seed=seed)
        return Response(result, status=status.HTTP_201_CREATED)
