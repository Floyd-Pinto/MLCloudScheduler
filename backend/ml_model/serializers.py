"""backend/ml_model/serializers.py"""
from rest_framework import serializers
from .models import ModelTrainingRun, ModelComparisonResult


class ModelTrainingRunSerializer(serializers.ModelSerializer):
    class Meta:
        model  = ModelTrainingRun
        fields = "__all__"


class ModelComparisonResultSerializer(serializers.ModelSerializer):
    class Meta:
        model  = ModelComparisonResult
        fields = "__all__"


class TrainRequestSerializer(serializers.Serializer):
    model_type = serializers.ChoiceField(
        choices=["gbr", "lstm", "arima", "combined", "all"],
        default="gbr",
    )


class PredictRequestSerializer(serializers.Serializer):
    history    = serializers.ListField(child=serializers.FloatField(), min_length=10)
    model_type = serializers.ChoiceField(
        choices=["gbr", "lstm", "arima", "combined"],
        default="gbr",
    )
