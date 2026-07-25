from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .views import (
    CreateDoctorView,
    DeactivateDoctorView,
    DeepSkinTokenObtainPairView,
    DoctorListView,
    MeView,
    RegisterPatientView,
)

urlpatterns = [
    path('register/', RegisterPatientView.as_view(), name='register-patient'),
    path('login/', DeepSkinTokenObtainPairView.as_view(), name='login'),
    path('login/refresh/', TokenRefreshView.as_view(), name='login-refresh'),
    path('me/', MeView.as_view(), name='me'),

    # Admin: doctor account management
    path('doctors/', DoctorListView.as_view(), name='doctor-list'),
    path('doctors/create/', CreateDoctorView.as_view(), name='doctor-create'),
    path('doctors/<int:pk>/deactivate/', DeactivateDoctorView.as_view(), name='doctor-deactivate'),
]
