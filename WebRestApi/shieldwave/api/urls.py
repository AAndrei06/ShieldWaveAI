from django.urls import path
from .views import UploadPredictionView, GetDeactivateInfo

urlpatterns = [
    path('upload/', UploadPredictionView.as_view(), name='upload_prediction'),
    path('deactivate/',GetDeactivateInfo.as_view(), name='deactivate_info')
]
