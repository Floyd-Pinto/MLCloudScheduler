"""config/urls.py — root URL configuration."""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/simulation/",  include("simulation.urls")),
    path("api/scheduler/",   include("scheduler.urls")),
    path("api/ml/",          include("ml_model.urls")),
    path("api/metrics/",     include("metrics.urls")),
    path("api/evaluation/",  include("evaluation.urls")),
]
