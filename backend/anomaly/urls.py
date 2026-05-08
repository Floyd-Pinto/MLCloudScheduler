"""anomaly/urls.py"""

from django.urls import path
from .views import AnomalyLogListView, AnomalyLogSummaryView

urlpatterns = [
    path("logs/",    AnomalyLogListView.as_view()),
    path("summary/", AnomalyLogSummaryView.as_view()),
]
