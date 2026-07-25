import uuid

from django.conf import settings
from django.db import models


def lesion_image_path(instance, filename):
    """Store uploads under media/cases/<case_uuid>/<filename> -- keeps each
    case's images grouped and avoids filename collisions across patients."""
    return f'cases/{instance.case.id}/{filename}'


class Case(models.Model):
    """
    One Case == one lesion a patient is concerned about. A patient with
    multiple separate moles creates multiple Cases (see system design doc,
    section 7) -- this keeps one prediction/verdict cleanly tied to one
    lesion rather than an ambiguous multi-lesion bundle.
    """

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'          # uploaded, AI processed, awaiting doctor pickup
        IN_REVIEW = 'in_review', 'In Review'    # a doctor has picked it up
        REVIEWED = 'reviewed', 'Reviewed'       # verdict recorded, patient notified

    class Priority(models.TextChoices):
        HIGH = 'high', 'High'
        MEDIUM = 'medium', 'Medium'
        LOW = 'low', 'Low'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='cases'
    )
    assigned_doctor = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='assigned_cases'
    )

    patient_note = models.TextField(blank=True, help_text="e.g. 'itchy for 2 weeks'")
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PENDING)

    # ── AI output (set once by the inference pipeline right after upload) ──
    # Doctors see the full fields below; patients never do (enforced in
    # serializers.py, not just the frontend -- see PatientCaseSerializer).
    ai_confidence = models.FloatField(null=True, blank=True)
    ai_priority = models.CharField(max_length=6, choices=Priority.choices, blank=True)
    attention_map_image = models.ImageField(upload_to='attention_maps/', null=True, blank=True)
    ai_processed_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    assigned_at = models.DateTimeField(null=True, blank=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'ai_confidence']),
            models.Index(fields=['patient', 'status']),
            models.Index(fields=['assigned_doctor', 'status']),
        ]

    def __str__(self):
        return f'Case {self.id} ({self.patient.username}, {self.status})'


class CaseImage(models.Model):
    """
    The original lesion photo(s) for a case. Modeled as its own table (not
    a single field on Case) so the patient can attach more than one angle
    of the SAME lesion if useful -- while the AI prediction/priority stays
    on the Case as a whole, computed from the primary image.
    """
    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to=lesion_image_path)
    is_primary = models.BooleanField(default=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-is_primary', 'uploaded_at']


class Verdict(models.Model):
    """
    The doctor's structured clinical decision for a case. Kept as an enum
    (not free text) so it stays queryable for audit/reporting -- the
    free-text `notes` field is for anything extra the doctor wants to add.
    One-to-one with Case: a case is reviewed once (re-review would create
    a new Case in this design, keeping history clean).
    """

    class Decision(models.TextChoices):
        REASSURE = 'reassure', 'Reassure -- no action needed'
        MONITOR = 'monitor', 'Monitor -- recheck in a few weeks'
        BIOPSY = 'biopsy', 'Recommend biopsy'
        REFER = 'refer', 'Refer to specialist'

    case = models.OneToOneField(Case, on_delete=models.CASCADE, related_name='verdict')
    doctor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT)
    decision = models.CharField(max_length=10, choices=Decision.choices)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class Message(models.Model):
    """
    Simple async message thread between a case's patient and doctor.
    Deliberately NOT real-time (no WebSocket) -- refresh-to-see-new-messages
    is sufficient for this use case, per the system design's explicit
    scope decision to skip real-time chat.
    """
    case = models.ForeignKey(Case, on_delete=models.CASCADE, related_name='messages')
    sender = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    body = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    read_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['created_at']
