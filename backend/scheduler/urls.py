"""scheduler/urls.py"""

from django.urls import path
from .views import (RunReactiveView, RunPredictiveView, RunComparisonView,
                    SchedulerRunListView, SchedulerRunDetailView)

urlpatterns = [
    path("reactive/",      RunReactiveView.as_view(),       name="scheduler-reactive"),
    path("predictive/",    RunPredictiveView.as_view(),     name="scheduler-predictive"),
    path("compare/",       RunComparisonView.as_view(),     name="scheduler-compare"),
    path("runs/",          SchedulerRunListView.as_view(),  name="scheduler-runs"),
    path("runs/<int:pk>/", SchedulerRunDetailView.as_view(),name="scheduler-run-detail"),
]
