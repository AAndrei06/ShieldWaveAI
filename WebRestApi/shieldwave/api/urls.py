from django.urls import path
from .views import UploadPredictionView, InitialCleanup, ActivityInfo, CheckUserExists, GetUserStatusInfo

urlpatterns = [
    path('upload/', UploadPredictionView.as_view(), name='upload_prediction'),

    path('get_status_info/',GetUserStatusInfo.as_view(), name='status_info'),

    path('activity_info/',ActivityInfo.as_view(), name="active_info"),
    path('initial_clean/',InitialCleanup.as_view(), name='initial_cleanup'),
    path('check_user/',CheckUserExists.as_view(), name="user_exists")
]
