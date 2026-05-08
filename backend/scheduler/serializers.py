"""scheduler/serializers.py — Updated for Phase 2 multi-resource fields."""

from rest_framework import serializers
from .models import SchedulerRun, SchedulerAction


class SchedulerActionSerializer(serializers.ModelSerializer):
    class Meta:
        model  = SchedulerAction
        fields = ["time_step", "workload", "capacity", "cpu_usage",
                  "memory_usage", "network_io", "overloaded", "action",
                  "trigger_resource"]


class SchedulerRunSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model  = SchedulerRun
        fields = ["id", "scheduler_type", "pattern", "steps", "seed",
                  "overload_events", "overload_rate", "avg_cpu", "avg_memory",
                  "avg_network", "avg_capacity", "total_cost",
                  "scale_up_count", "scale_down_count",
                  "overload_cpu_count", "overload_memory_count",
                  "overload_network_count", "created_at"]


class SchedulerRunDetailSerializer(serializers.ModelSerializer):
    actions = SchedulerActionSerializer(many=True, read_only=True)

    class Meta:
        model  = SchedulerRun
        fields = ["id", "scheduler_type", "pattern", "steps", "seed",
                  "overload_events", "overload_rate", "avg_cpu", "avg_memory",
                  "avg_network", "avg_capacity", "total_cost",
                  "scale_up_count", "scale_down_count",
                  "overload_cpu_count", "overload_memory_count",
                  "overload_network_count", "created_at", "actions"]


class RunSchedulerSerializer(serializers.Serializer):
    pattern = serializers.ChoiceField(choices=["gradual", "spike", "periodic", "combined"], default="combined")
    steps   = serializers.IntegerField(min_value=50, max_value=1000, default=200)
    seed    = serializers.IntegerField(min_value=0, max_value=99999, default=42)
