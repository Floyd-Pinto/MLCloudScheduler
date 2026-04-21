"""evaluation/serializers.py"""

from rest_framework import serializers
from .models import EvaluationResult


class EvaluationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model  = EvaluationResult
        fields = "__all__"


class RunEvaluationSerializer(serializers.Serializer):
    pattern = serializers.ChoiceField(choices=["gradual", "spike", "periodic", "combined"], default="combined")
    steps   = serializers.IntegerField(min_value=50, max_value=1000, default=200)
    seed    = serializers.IntegerField(min_value=0, max_value=99999, default=42)
