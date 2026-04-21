"""evaluation/urls.py"""

from django.urls import path
from .views import RunEvaluationView, EvaluationListView, EvaluationComparisonView

urlpatterns = [
    path("run/",        RunEvaluationView.as_view(),      name="evaluation-run"),
    path("",            EvaluationListView.as_view(),     name="evaluation-list"),
    path("comparison/", EvaluationComparisonView.as_view(),name="evaluation-comparison"),
]
