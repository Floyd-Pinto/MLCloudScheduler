"""simulation/serializers.py"""

from rest_framework import serializers
from .models import WorkloadRun, WorkloadDataPoint


class WorkloadDataPointSerializer(serializers.ModelSerializer):
    class Meta:
        model  = WorkloadDataPoint
        fields = ["time_step", "workload"]


class WorkloadRunSerializer(serializers.ModelSerializer):
    datapoints = WorkloadDataPointSerializer(many=True, read_only=True)

    class Meta:
        model  = WorkloadRun
        fields = ["id", "pattern", "steps", "seed", "label", "created_at", "datapoints"]


class WorkloadRunListSerializer(serializers.ModelSerializer):
    class Meta:
        model  = WorkloadRun
        fields = ["id", "pattern", "steps", "seed", "label", "created_at"]


class GenerateWorkloadSerializer(serializers.Serializer):
    pattern = serializers.ChoiceField(choices=["gradual", "spike", "periodic", "combined"],
                                      default="combined")
    steps   = serializers.IntegerField(min_value=50, max_value=1000, default=200)
    seed    = serializers.IntegerField(min_value=0, max_value=99999, default=42)
    label   = serializers.CharField(max_length=120, required=False, allow_blank=True)
