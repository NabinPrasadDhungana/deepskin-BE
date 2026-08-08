"""
Celery task that hands a new case to the hosted inference Space.

cases/views.py schedules this with .delay(case.id) instead of calling the
pipeline directly -- keeps the upload endpoint fast and isolates inference
failures from the request lifecycle entirely.

Flow (async callback design):
    1. worker marks case ai_status=PROCESSING
    2. for each CaseImage, POST its bytes to the HF Space (gets 202)
    3. HF runs the model and POSTs {client_ref, confidence, prediction,
       attention_png_base64} back to the Django webhook endpoint
    4. the webhook (mlservice/views.py ResultWebhookView) writes per-image
       results, rolls the worst up to the Case and marks it DONE/FAILED.
"""
from celery import shared_task

from notifications.emitter import emit_notification, emit_notification_to_admins
from notifications.models import Notification

from .inference_client import submit_image


@shared_task(bind=True, max_retries=2)
def process_case_task(self, case_id):
    from cases.models import Case

    try:
        case = Case.objects.get(pk=case_id)
    except Case.DoesNotExist:
        return  # case was deleted before the task ran -- nothing to do

    case.ai_status = Case.AIStatus.PROCESSING
    case.ai_error_message = ''
    case.save(update_fields=['ai_status', 'ai_error_message'])

    try:
        for case_image in case.images.all():
            # Read original bytes from the storage backend (B2 in prod,
            # local disk in dev) and hand them to HF.
            with case_image.image.open('rb') as fh:
                image_bytes = fh.read()
            submit_image(case.id, case_image.id, image_bytes)
    except Exception as exc:
        case.ai_status = Case.AIStatus.FAILED
        case.ai_error_message = str(exc)[:500]
        case.save(update_fields=['ai_status', 'ai_error_message'])

        emit_notification_to_admins(
            type_=Notification.Type.AI_FAILED,
            body=f'AI processing failed for case {case.pk}. Check the admin audit log.',
            actor='System',
            case=case,
        )
        # Retry once or twice (transient HF space cold-start / reachability),
        # then give up and leave it FAILED for an admin to notice.
        raise self.retry(exc=exc, countdown=30)