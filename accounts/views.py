from django.utils import timezone

from rest_framework import generics, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.parsers import MultiPartParser, FormParser

from .models import User
from .permissions import IsAdminRole
from .serializers import (
    CreateDoctorSerializer,
    DeepSkinTokenObtainPairSerializer,
    DoctorApplicationSerializer,
    DoctorSelfRegisterSerializer,
    RegisterPatientSerializer,
    ReviewDoctorApplicationSerializer,
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

    queryset = User.objects.filter(role=User.Role.DOCTOR).order_by("username")
    serializer_class = UserSerializer
    permission_classes = [IsAdminRole]


class DeactivateDoctorView(APIView):
    """POST /api/auth/doctors/<id>/deactivate/ -- Admin only."""

    permission_classes = [IsAdminRole]

    def post(self, request, pk):
        try:
            doctor = User.objects.get(pk=pk, role=User.Role.DOCTOR)
        except User.DoesNotExist:
            return Response({"detail": "Doctor not found."}, status=404)
        doctor.is_active = False
        doctor.save(update_fields=["is_active"])
        return Response({"detail": f"Doctor {doctor.username} deactivated."})


class MeView(APIView):
    """GET /api/auth/me/ -- any authenticated user's own profile."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        return Response(UserSerializer(request.user).data)


class DeepSkinTokenObtainPairView(TokenObtainPairView):
    """POST /api/auth/login/ -- login, returns access+refresh tokens with role embedded."""

    serializer_class = DeepSkinTokenObtainPairSerializer


class DoctorSelfRegisterView(generics.CreateAPIView):
    """POST /api/auth/doctors/register/ -- public, multipart (license_document file).
    Creates an inactive, PENDING doctor account -- cannot log in until approved."""

    queryset = User.objects.all()
    serializer_class = DoctorSelfRegisterSerializer
    permission_classes = [permissions.AllowAny]
    parser_classes = [MultiPartParser, FormParser]


class PendingDoctorApplicationsView(generics.ListAPIView):
    """GET /api/auth/doctors/pending/ -- Admin only."""

    serializer_class = DoctorApplicationSerializer
    permission_classes = [IsAdminRole]

    def get_queryset(self):
        return User.objects.filter(
            role=User.Role.DOCTOR,
            verification_status=User.VerificationStatus.PENDING,
        ).order_by("date_joined")


class ReviewDoctorApplicationView(APIView):
    """
    POST /api/auth/doctors/<id>/review/  { "decision": "approve"|"reject", "notes": "..." }
    Admin only. Approving flips is_active=True so the doctor can log in;
    rejecting keeps them locked out and records the reason.
    """

    permission_classes = [IsAdminRole]

    def post(self, request, pk):
        try:
            doctor = User.objects.get(pk=pk, role=User.Role.DOCTOR)
        except User.DoesNotExist:
            return Response({"detail": "Doctor application not found."}, status=404)

        if doctor.verification_status != User.VerificationStatus.PENDING:
            return Response(
                {"detail": "This application has already been reviewed."}, status=409
            )

        serializer = ReviewDoctorApplicationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        decision = serializer.validated_data["decision"]
        doctor.verification_notes = serializer.validated_data.get("notes", "")
        doctor.reviewed_by = request.user
        doctor.reviewed_at = timezone.now()

        if decision == "approve":
            doctor.verification_status = User.VerificationStatus.APPROVED
            doctor.is_active = True
        else:
            doctor.verification_status = User.VerificationStatus.REJECTED
            doctor.is_active = False

        doctor.save(
            update_fields=[
                "verification_status",
                "is_active",
                "verification_notes",
                "reviewed_by",
                "reviewed_at",
            ]
        )
        return Response(DoctorApplicationSerializer(doctor).data)
