"""
Webhook endpoint that receives async inference results from the HF Space.

The web (gunicorn) process gets POST /api/ml/result/ {client_ref, case_id,
image_id, confidence, prediction, attention_png_base64} for each image.
It persists the per-image result and, once the last image of a case has
landed, rolls the worst-case (highest malignant score) image up to the
Case and emits the CASE_READY notification (-> SSE bell).

Idempotent: results keyed by CaseImage are safe to re-deliver (HF may retry
a callback, and the worker can resubmit on retry).

Auth: shared secret X-HF-Token (DEEPSKIN_HF_TOKEN). DRF auth is disabled
here because HF talks server-to-server, not as an end user.
"""
import base64
import logging

from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from cases.models import Case, CaseImage
from notifications.emitter import emit_notification, emit_notification_to_admins
from notifications.models import Notification

from .pipeline import confidence_to_priority

logger = logging.getLogger('deepskin.hf_webhook')


class ResultWebhookView(APIView):
    """POST /api/ml/result/ -- inference result callback from the HF Space."""

    authentication_classes = []
    permission_classes = []

    def post(self, request):
        # Shared-secret auth with the HF Space. Enforced only when a token
        # is configured, so local dev (no env vars) still works end-to-end.
        expected = settings.DEEPSKIN_HF_TOKEN
        if expected and request.headers.get('X-HF-Token', '') != expected:
            return Response({'detail': 'Forbidden'}, status=status.HTTP_403_FORBIDDEN)

        data = request.data
        case_id = data.get('case_id')
        image_id = data.get('image_id')
        error = data.get('error')

        try:
            case = Case.objects.get(pk=case_id)
            case_image = CaseImage.objects.get(pk=image_id, case=case)
        except (Case.DoesNotExist, CaseImage.DoesNotExist):
            return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)

        if error:
            self._fail_case(case, error)
            return Response({'detail': 'recorded failure'}, status=status.HTTP_200_OK)

        try:
            confidence = float(data['confidence'])
            prediction = data.get('prediction') or (
                Case.AIPrediction.MALIGNANT
                if confidence >= settings.DEEPSKIN_CLASSIFICATION_THRESHOLD
                else Case.AIPrediction.BENIGN
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._fail_case(case, f'Malformed webhook payload: {exc}')
            return Response({'detail': 'recorded failure'}, status=status.HTTP_200_OK)

        overlay_bytes = None
        raw_overlay = data.get('attention_png_base64')
        if raw_overlay:
            try:
                overlay_bytes = base64.b64decode(raw_overlay)
            except Exception:
                logger.warning('bad attention_png_base64 for image %s', image_id)

        now = timezone.now()
        case_image.ai_confidence = confidence
        case_image.ai_prediction = prediction
        case_image.ai_processed_at = now
        if overlay_bytes:
            case_image.ai_attention_map.save(
                f'{case.id}_{case_image.pk}_attention.png',
                ContentFile(overlay_bytes), save=False,
            )
        case_image.save()

        self._maybe_complete_case(case)
        return Response({'detail': 'ok'}, status=status.HTTP_200_OK)

    # ── helpers ──────────────────────────────────────────────────────────

    def _maybe_complete_case(self, case):
        """Once every image of the case has a result, do the worst-case
        roll-up, mark the case DONE and notify the patient."""
        pending = case.images.filter(ai_processed_at__isnull=True)
        if pending.exists():
            return  # still waiting on other images

        results = list(case.images.all())
        worst = max(results, key=lambda img: img.ai_confidence or -1.0)
        worst_conf = worst.ai_confidence or 0.0

        case.ai_confidence = worst_conf
        case.ai_prediction = (
            Case.AIPrediction.MALIGNANT
            if worst_conf >= settings.DEEPSKIN_CLASSIFICATION_THRESHOLD
            else Case.AIPrediction.BENIGN
        )
        case.ai_priority = confidence_to_priority(worst_conf)

        if worst.ai_attention_map:
            worst.ai_attention_map.open('rb')
            case.attention_map_image.save(
                f'{case.id}_attention.png',
                ContentFile(worst.ai_attention_map.read()), save=False,
            )
            worst.ai_attention_map.close()

        case.ai_status = Case.AIStatus.DONE
        case.ai_processed_at = timezone.now()
        case.save(update_fields=[
            'ai_confidence', 'ai_prediction', 'ai_priority',
            'attention_map_image', 'ai_status', 'ai_processed_at',
        ])

        emit_notification(
            recipient=case.patient,
            type_=Notification.Type.CASE_READY,
            body='Your case is ready and will be reviewed by a doctor soon.',
            actor='System',
            case=case,
        )

    def _fail_case(self, case, error):
        case.ai_status = Case.AIStatus.FAILED
        case.ai_error_message = str(error)[:500]
        case.save(update_fields=['ai_status', 'ai_error_message'])
        emit_notification_to_admins(
            type_=Notification.Type.AI_FAILED,
            body=f'AI processing failed for case {case.pk}. Check the admin audit log.',
            actor='System',
            case=case,
        )