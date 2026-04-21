"""metrics/serializers.py"""

from rest_framework import serializers
from .models import MetricRecord


class MetricRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model  = MetricRecord
        fields = ["id", "scheduler_type", "pattern", "steps",
                  "overload_events", "overload_rate", "avg_cpu", "avg_capacity",
                  "total_cost", "scale_up_count", "scale_down_count", "recorded_at"]
