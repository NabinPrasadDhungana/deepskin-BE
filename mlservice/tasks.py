"""
Celery task wrapping the ML pipeline. cases/views.py schedules this with
.delay(case.id) instead of calling run_inference_on_case() directly --
keeps the upload endpoint fast and isolates ML failures from the request
lifecycle entirely (a model crash here just marks ai_status=FAILED rather
than 500-ing the patient's upload request, which already succeeded).
"""
from celery import shared_task
from django.utils import timezone

from notifications.emitter import emit_notification, emit_notification_to_admins
from notifications.models import Notification


@shared_task(bind=True, max_retries=2)
def process_case_task(self, case_id):
    from cases.models import Case
    from .pipeline import run_inference_on_case

    try:
        case = Case.objects.get(pk=case_id)
    except Case.DoesNotExist:
        return  # case was deleted before the task ran -- nothing to do

    case.ai_status = Case.AIStatus.PROCESSING
    case.save(update_fields=['ai_status'])

    try:
        run_inference_on_case(case)
        case.ai_status = Case.AIStatus.DONE
        case.save(update_fields=['ai_status'])

        emit_notification(
            recipient=case.patient,
            type_=Notification.Type.CASE_READY,
            body='Your case is ready and will be reviewed by a doctor soon.',
            actor='System',
            case=case,
        )
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
        # Retry once or twice (transient GPU/model load issues), then give
        # up and leave it FAILED for an admin to notice via admin/stats.
        raise self.retry(exc=exc, countdown=30)