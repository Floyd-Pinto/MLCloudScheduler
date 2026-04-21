"""simulation/urls.py"""

from django.urls import path
from .views import GenerateWorkloadView, WorkloadRunListView, WorkloadRunDetailView

urlpatterns = [
    path("generate/",     GenerateWorkloadView.as_view(),   name="simulation-generate"),
    path("runs/",         WorkloadRunListView.as_view(),    name="simulation-runs"),
    path("runs/<int:pk>/", WorkloadRunDetailView.as_view(), name="simulation-run-detail"),
]
