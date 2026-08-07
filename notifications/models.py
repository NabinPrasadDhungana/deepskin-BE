from django.conf import settings
from django.db import models


class Notification(models.Model):
    """
    An in-app notification delivered to a single user.

    The row is the source of truth: it persists regardless of whether the
    real-time push (SSE via Redis pub/sub, see emitter.py) succeeds. If the
    push fails, the frontend's initial-load / list endpoint still surfaces
    the notification.

    Type enum mirrors the domain events that generate notifications. Body
    text is what the user sees in the bell dropdown. SAFETY: patient-facing
    bodies must never reference AI confidence/priority/attention maps --
    same rule as the patient case serializers.
    """

    class Type(models.TextChoices):
        CASE_SUBMITTED = 'case_submitted', 'Case submitted'
        CASE_READY = 'case_ready', 'Case ready for a doctor'
        PICKUP = 'pickup', 'Case picked up'
        VERDICT = 'verdict', 'Verdict submitted'
        MESSAGE = 'message', 'New message'
        APPLICATION = 'application', 'Doctor application'
        AI_FAILED = 'ai_failed', 'AI processing failed'

    recipient = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='notifications'
    )
    type = models.CharField(max_length=20, choices=Type.choices)
    body = models.TextField()
    
    
    actor = models.CharField(max_length=150, blank=True)
    case = models.ForeignKey(
        'cases.Case', on_delete=models.CASCADE, null=True, blank=True,
        related_name='notifications',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    read_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['recipient', 'read_at']),
            models.Index(fields=['recipient', '-created_at']),
        ]

    def __str__(self):
        return f'{self.type} -> {self.recipient.username} ({self.created_at:%Y-%m-%d %H:%M})'
