"""anomaly/serializers.py"""

from rest_framework import serializers
from .models import AnomalyLogEntry


class AnomalyLogEntrySerializer(serializers.ModelSerializer):
    class Meta:
        model  = AnomalyLogEntry
        fields = "__all__"
