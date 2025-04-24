from django.urls import path
from .views import UploadPredictionView, GetDeactivateInfo, GetCameraDeactivateInfo, GetMicDeactivateInfo, \
InitialCleanup, GetActivateInfo, ActivityInfo, CheckUserExists

urlpatterns = [
    path('upload/', UploadPredictionView.as_view(), name='upload_prediction'),
    path('deactivate/',GetDeactivateInfo.as_view(), name='deactivate_info'),
    path('deactivate_cam/',GetCameraDeactivateInfo.as_view(), name='deactivate_cam'),
    path('deactivate_mic/',GetMicDeactivateInfo.as_view(), name='deactivate_mic'),
    path('initial_clean/',InitialCleanup.as_view(), name='initial_cleanup'),
    path('activate/',GetActivateInfo.as_view(), name='activate'),
    path('activity_info/',ActivityInfo.as_view(), name="active_info"),
    path('check_user/',CheckUserExists.as_view(), name="user_exists")
]
