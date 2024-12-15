from django.urls import path
from .views import UploadPredictionView

urlpatterns = [
    path('upload/', UploadPredictionView.as_view(), name='upload_prediction'),
]
