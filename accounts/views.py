from rest_framework import generics, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView

from .models import User
from .permissions import IsAdminRole
from .serializers import (
    CreateDoctorSerializer,
    DeepSkinTokenObtainPairSerializer,
    RegisterPatientSerializer,
    UserSerializer,
)


class RegisterPatientView(generics.CreateAPIView):
    """POST /api/auth/register/ -- public, always creates a Patient account."""
    queryset = User.objects.all()
    serializer_class = RegisterPatientSerializer
    permission_classes = [permissions.AllowAny]


class CreateDoctorView(generics.CreateAPIView):
    """
    POST /api/auth/doctors/ -- Admin only.
    Doctor accounts are deliberately NOT self-service: an Admin provisions
    them, matching the "Admin manages doctor accounts" responsibility from
    the system design.
    """
    queryset = User.objects.all()
    serializer_class = CreateDoctorSerializer
    permission_classes = [IsAdminRole]


class DoctorListView(generics.ListAPIView):
    """GET /api/auth/doctors/ -- Admin only. For the Doctor Management screen."""
    queryset = User.objects.filter(role=User.Role.DOCTOR).order_by('username')
    serializer_class = UserSerializer
    permission_classes = [IsAdminRole]


class DeactivateDoctorView(APIView):
    """POST /api/auth/doctors/<id>/deactivate/ -- Admin only."""
    permission_classes = [IsAdminRole]

    def post(self, request, pk):
        try:
            doctor = User.objects.get(pk=pk, role=User.Role.DOCTOR)
        except User.DoesNotExist:
            return Response({'detail': 'Doctor not found.'}, status=404)
        doctor.is_active = False
        doctor.save(update_fields=['is_active'])
        return Response({'detail': f'Doctor {doctor.username} deactivated.'})


class MeView(APIView):
    """GET /api/auth/me/ -- any authenticated user's own profile."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        return Response(UserSerializer(request.user).data)


class DeepSkinTokenObtainPairView(TokenObtainPairView):
    """POST /api/auth/login/ -- login, returns access+refresh tokens with role embedded."""
    serializer_class = DeepSkinTokenObtainPairSerializer
