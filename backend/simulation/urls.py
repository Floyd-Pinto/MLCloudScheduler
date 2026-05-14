"""simulation/urls.py"""

from django.urls import path
from .views import GenerateWorkloadView, WorkloadRunListView, WorkloadRunDetailView, TracePreviewView

urlpatterns = [
    path("generate/",     GenerateWorkloadView.as_view(),   name="simulation-generate"),
    path("runs/",         WorkloadRunListView.as_view(),    name="simulation-runs"),
    path("runs/<int:pk>/", WorkloadRunDetailView.as_view(), name="simulation-run-detail"),
    path("trace-preview/", TracePreviewView.as_view(),      name="simulation-trace-preview"),
]
