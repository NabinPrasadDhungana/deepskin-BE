from rest_framework import serializers

from accounts.serializers import UserSerializer
from .models import Case, CaseImage, Message, Verdict


class CaseImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = CaseImage
        fields = ['id', 'image', 'is_primary', 'uploaded_at']
        read_only_fields = ['id', 'uploaded_at']


class VerdictSerializer(serializers.ModelSerializer):
    doctor = UserSerializer(read_only=True)

    class Meta:
        model = Verdict
        fields = ['id', 'decision', 'notes', 'doctor', 'created_at']
        read_only_fields = ['id', 'doctor', 'created_at']


class MessageSerializer(serializers.ModelSerializer):
    sender = UserSerializer(read_only=True)

    class Meta:
        model = Message
        fields = ['id', 'sender', 'body', 'created_at', 'read_at']
        read_only_fields = ['id', 'sender', 'created_at', 'read_at']


# ─────────────────────────────────────────────────────────────────────────
# Patient-facing serializers
# HARD RULE: these must never expose ai_confidence, ai_priority, or
# attention_map_image. This is the code-level enforcement of the "patients
# never see the AI's raw output" constraint from the system design -- not
# just a frontend convention.
# ─────────────────────────────────────────────────────────────────────────

class PatientCaseListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Case
        fields = ['id', 'status', 'patient_note', 'created_at', 'reviewed_at']
        read_only_fields = fields


class PatientCaseDetailSerializer(serializers.ModelSerializer):
    images = CaseImageSerializer(many=True, read_only=True)
    verdict = VerdictSerializer(read_only=True)
    assigned_doctor = UserSerializer(read_only=True)

    class Meta:
        model = Case
        fields = [
            'id', 'status', 'patient_note', 'images',
            'assigned_doctor', 'verdict', 'created_at', 'reviewed_at',
        ]
        read_only_fields = fields


class CreateCaseSerializer(serializers.ModelSerializer):
    """Used for the patient's initial upload. Images are handled separately
    in the view (multipart, possibly multiple files) -- see cases/views.py."""

    class Meta:
        model = Case
        fields = ['id', 'patient_note']
        read_only_fields = ['id']


# ─────────────────────────────────────────────────────────────────────────
# Doctor-facing serializers
# ─────────────────────────────────────────────────────────────────────────

class DoctorQueueSerializer(serializers.ModelSerializer):
    """
    For the unassigned-case queue list. Deliberately coarse: shows the
    priority BUCKET (High/Medium/Low), not the exact confidence number --
    see system design doc section 4 for why (avoid over-anchoring doctors
    on a raw score before they've looked at the case themselves).
    """
    thumbnail = serializers.SerializerMethodField()

    class Meta:
        model = Case
        fields = ['id', 'ai_priority', 'thumbnail', 'created_at']
        read_only_fields = fields

    def get_thumbnail(self, obj):
        primary = obj.images.filter(is_primary=True).first() or obj.images.first()
        if not primary:
            return None
        request = self.context.get('request')
        url = primary.image.url
        return request.build_absolute_uri(url) if request else url


class DoctorCaseDetailSerializer(serializers.ModelSerializer):
    """
    Full detail view -- shown only once a doctor has picked up the case
    (enforced by IsAssignedDoctorOrUnassigned in the view). Includes the
    exact AI confidence and the attention-map overlay.
    """
    images = CaseImageSerializer(many=True, read_only=True)
    verdict = VerdictSerializer(read_only=True)
    patient = UserSerializer(read_only=True)

    class Meta:
        model = Case
        fields = [
            'id', 'patient', 'patient_note', 'status',
            'images', 'ai_confidence', 'ai_priority', 'attention_map_image',
            'verdict', 'created_at', 'assigned_at', 'reviewed_at',
        ]
        read_only_fields = fields


class RecordVerdictSerializer(serializers.ModelSerializer):
    class Meta:
        model = Verdict
        fields = ['decision', 'notes']
