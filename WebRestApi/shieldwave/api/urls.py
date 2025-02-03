from django.urls import path
from .views import UploadPredictionView, GetDeactivateInfo, GetCameraDeactivateInfo, GetMicDeactivateInfo, InitialCleanup

urlpatterns = [
    path('upload/', UploadPredictionView.as_view(), name='upload_prediction'),
    path('deactivate/',GetDeactivateInfo.as_view(), name='deactivate_info'),
    path('deactivate_cam/',GetCameraDeactivateInfo.as_view(), name='deactivate_cam'),
    path('deactivate_mic/',GetMicDeactivateInfo.as_view(), name='deactivate_mic'),
    path('initial_clean/',InitialCleanup.as_view(), name='initial_cleanup')
]
