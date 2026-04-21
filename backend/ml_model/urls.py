"""backend/ml_model/urls.py"""
from django.urls import path
from . import views

urlpatterns = [
    path("train/",           views.TrainView.as_view()),
    path("status/",          views.ModelStatusView.as_view()),
    path("history/",         views.ModelHistoryView.as_view()),
    path("predict/",         views.PredictView.as_view()),
    path("predict-all/",     views.PredictAllView.as_view()),
    path("compare-models/",  views.ModelCompareView.as_view()),
]
