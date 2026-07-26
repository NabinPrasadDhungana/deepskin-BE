from django.contrib.auth import get_user_model
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import generics, permissions, status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from accounts.permissions import IsDoctor, IsPatient
from mlservice.tasks import process_case_task
from .models import Case, CaseImage, Message
from .permissions import IsAssignedDoctorOrUnassigned, IsCaseOwnerPatient
from .serializers import (
    CreateCaseSerializer,
    DoctorCaseDetailSerializer,
    DoctorQueueSerializer,
    MessageSerializer,
    PatientCaseDetailSerializer,
    PatientCaseListSerializer,
    RecordVerdictSerializer,
)

User = get_user_model()


# ─────────────────────────────────────────────────────────────────────────
# Patient endpoints
# ─────────────────────────────────────────────────────────────────────────

class PatientCaseListCreateView(generics.ListCreateAPIView):
    """
    GET  /api/cases/mine/  -- patient's own case list
    POST /api/cases/mine/  -- upload a new case (multipart: note + image(s))

    NOTE ON scope: ML inference is run synchronously here for simplicity.
    In a production deployment this should be pushed to a background task
    queue (e.g. Celery) so the upload request returns immediately and the
    case appears in the doctor queue once processing finishes -- flagged
    here rather than silently left as a scaling gap.
    """
    permission_classes = [IsPatient]
    parser_classes = [MultiPartParser, FormParser]

    def get_queryset(self):
        return Case.objects.filter(patient=self.request.user)

    def get_serializer_class(self):
        return CreateCaseSerializer if self.request.method == 'POST' else PatientCaseListSerializer

    def create(self, request, *args, **kwargs):
        images = request.FILES.getlist('images')
        if not images:
            return Response({'detail': 'At least one image is required.'}, status=400)

        case = Case.objects.create(
            patient=request.user,
            patient_note=request.data.get('patient_note', ''),
        )
        for i, img in enumerate(images):
            CaseImage.objects.create(case=case, image=img, is_primary=(i == 0))

        # Schedule async -- returns immediately, doctor queue fills in
        # once the task finishes (ai_status: queued -> processing -> done).
        process_case_task.delay(str(case.id))

        return Response(
            PatientCaseDetailSerializer(case, context={'request': request}).data,
            status=status.HTTP_201_CREATED,
        )


class PatientCaseDetailView(generics.RetrieveAPIView):
    """GET /api/cases/mine/<id>/ -- patient view, AI fields never included."""
    serializer_class = PatientCaseDetailSerializer
    permission_classes = [IsPatient, IsCaseOwnerPatient]
    queryset = Case.objects.all()


# ─────────────────────────────────────────────────────────────────────────
# Doctor endpoints
# ─────────────────────────────────────────────────────────────────────────

class DoctorQueueView(generics.ListAPIView):
    """
    GET /api/cases/queue/ -- unassigned cases, highest AI priority first.
    Coarse priority bucket only (see DoctorQueueSerializer docstring).
    """
    serializer_class = DoctorQueueSerializer
    permission_classes = [IsDoctor]

    def get_queryset(self):
        return (
            Case.objects.filter(
                status=Case.Status.PENDING,
                assigned_doctor__isnull=True,
                ai_status=Case.AIStatus.DONE,
            )
            .order_by('-ai_confidence', 'created_at')
        )


class DoctorMyCasesView(generics.ListAPIView):
    """GET /api/cases/mine-as-doctor/ -- cases this doctor has picked up."""
    serializer_class = DoctorCaseDetailSerializer
    permission_classes = [IsDoctor]

    def get_queryset(self):
        return Case.objects.filter(assigned_doctor=self.request.user).order_by('-assigned_at')


class DoctorCaseDetailView(generics.RetrieveAPIView):
    """GET /api/cases/<id>/detail/ -- full detail incl. AI confidence + attention map."""
    serializer_class = DoctorCaseDetailSerializer
    permission_classes = [IsDoctor, IsAssignedDoctorOrUnassigned]
    queryset = Case.objects.all()


class PickUpCaseView(APIView):
    """POST /api/cases/<id>/pickup/ -- doctor self-assigns an unassigned case."""
    permission_classes = [IsDoctor]

    def post(self, request, pk):
        case = get_object_or_404(Case, pk=pk)
        if case.assigned_doctor_id is not None:
            return Response({'detail': 'Case is already assigned.'}, status=409)
        case.assigned_doctor = request.user
        case.status = Case.Status.IN_REVIEW
        case.assigned_at = timezone.now()
        case.save(update_fields=['assigned_doctor', 'status', 'assigned_at'])
        return Response(DoctorCaseDetailSerializer(case, context={'request': request}).data)


class RecordVerdictView(generics.CreateAPIView):
    """
    POST /api/cases/<id>/verdict/ -- doctor's structured decision.
    Only the doctor already assigned to the case may record a verdict.
    Marks the case Reviewed and stamps reviewed_at.
    """
    serializer_class = RecordVerdictSerializer
    permission_classes = [IsDoctor]

    def create(self, request, *args, **kwargs):
        case = get_object_or_404(Case, pk=kwargs['pk'])
        if case.assigned_doctor_id != request.user.id:
            return Response({'detail': 'You are not assigned to this case.'}, status=403)
        if hasattr(case, 'verdict'):
            return Response({'detail': 'This case already has a verdict.'}, status=409)

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(case=case, doctor=request.user)

        case.status = Case.Status.REVIEWED
        case.reviewed_at = timezone.now()
        case.save(update_fields=['status', 'reviewed_at'])

        return Response(serializer.data, status=status.HTTP_201_CREATED)


class PatientHistoryForDoctorView(generics.ListAPIView):
    """
    GET /api/cases/patient-history/<patient_id>/ -- a given patient's past
    REVIEWED cases, for the doctor's "patient history" panel (system design
    section 5, item 10). Only reviewed cases are shown -- a doctor should
    not see another patient's cases still pending/unassigned via this route.
    """
    serializer_class = DoctorCaseDetailSerializer
    permission_classes = [IsDoctor]

    def get_queryset(self):
        return Case.objects.filter(
            patient_id=self.kwargs['patient_id'],
            status=Case.Status.REVIEWED,
        ).order_by('-reviewed_at')


# ─────────────────────────────────────────────────────────────────────────
# Messaging (shared between patient and assigned doctor on a case)
# ─────────────────────────────────────────────────────────────────────────

class CaseMessageListCreateView(generics.ListCreateAPIView):
    """
    GET/POST /api/cases/<id>/messages/
    Accessible to the case's patient and its assigned doctor only.
    Simple polling-based thread, not real-time (see Message model docstring).
    """
    serializer_class = MessageSerializer
    permission_classes = [permissions.IsAuthenticated]

    def _get_case_or_403(self):
        case = get_object_or_404(Case, pk=self.kwargs['pk'])
        user = self.request.user
        allowed = (
            (user.is_patient() and case.patient_id == user.id)
            or (user.is_doctor() and case.assigned_doctor_id == user.id)
            or user.is_admin_role()
        )
        if not allowed:
            return None
        return case

    def get_queryset(self):
        case = self._get_case_or_403()
        if case is None:
            return Message.objects.none()
        return case.messages.all()

    def create(self, request, *args, **kwargs):
        case = self._get_case_or_403()
        if case is None:
            return Response({'detail': 'Not authorized for this case.'}, status=403)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(case=case, sender=request.user)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


# ─────────────────────────────────────────────────────────────────────────
# Admin endpoints (system design doc section 2.3 / screen 11-13)
# ─────────────────────────────────────────────────────────────────────────

class AdminCaseAuditListView(generics.ListAPIView):
    """
    GET /api/cases/admin/audit/ -- every case, full detail, for the
    Admin Audit Log screen. Admins see everything; this is the one place
    that's intentionally exempt from the per-doctor case isolation rules.
    """
    serializer_class = DoctorCaseDetailSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if not self.request.user.is_admin_role():
            return Case.objects.none()
        return Case.objects.all().order_by('-created_at')


class AdminStatsView(APIView):
    """
    GET /api/cases/admin/stats/ -- summary numbers for the Admin Dashboard
    screen: queue length, average time-to-review, case volume. Deliberately
    a plain aggregate endpoint rather than a generic analytics framework --
    matches the scope of what the design doc actually asks for.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        if not request.user.is_admin_role():
            return Response({'detail': 'Admin access required.'}, status=403)

        from django.db.models import Avg, F

        total_cases = Case.objects.count()
        queue_length = Case.objects.filter(
            status=Case.Status.PENDING, assigned_doctor__isnull=True
        ).count()
        in_review_count = Case.objects.filter(status=Case.Status.IN_REVIEW).count()
        reviewed_count = Case.objects.filter(status=Case.Status.REVIEWED).count()

        avg_time_to_review = (
            Case.objects.filter(status=Case.Status.REVIEWED, reviewed_at__isnull=False)
            .annotate(turnaround=F('reviewed_at') - F('created_at'))
            .aggregate(avg=Avg('turnaround'))['avg']
        )

        return Response({
            'total_cases': total_cases,
            'queue_length': queue_length,
            'in_review_count': in_review_count,
            'reviewed_count': reviewed_count,
            'avg_time_to_review_seconds': (
                avg_time_to_review.total_seconds() if avg_time_to_review else None
            ),
            'active_doctor_count': User.objects.filter(
                role=User.Role.DOCTOR, is_active=True
            ).count(),
        })
