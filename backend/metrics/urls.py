"""metrics/urls.py"""

from django.urls import path
from .views import MetricListView, MetricSummaryView

urlpatterns = [
    path("",         MetricListView.as_view(),   name="metrics-list"),
    path("summary/", MetricSummaryView.as_view(),name="metrics-summary"),
]
