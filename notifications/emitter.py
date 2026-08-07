"""
Notification delivery helpers.

A notification is persisted as a Notification row (always) and then pushed
in real time over Redis pub/sub (best-effort). The SSE stream endpoint in
views.py subscribes to `notify:<user_id>` channels, so producers in the
Celery worker process can reach SSE connections held by the Django server
process without any in-process event hub.
"""
import json

import redis
from django.conf import settings
from django.core.serializers.json import DjangoJSONEncoder

from .models import Notification
from .serializers import NotificationSerializer


def _publish(recipient_id, payload):
    """Publish to Redis; swallow failures so a down Redis never blocks the
    caller (the persisted row is the fallback for the frontend's list call)."""
    try:
        r = redis.Redis.from_url(settings.CELERY_BROKER_URL)
        try:
            
            
            r.publish(f'notify:{recipient_id}', json.dumps(payload, cls=DjangoJSONEncoder))
        finally:
            r.close()
    except Exception:
        pass


def emit_notification(*, recipient, type_, body, actor='', case=None):
    """Create a Notification row and push it live. Returns the instance."""
    notification = Notification.objects.create(
        recipient=recipient, type=type_, body=body, actor=actor, case=case,
    )
    _publish(recipient.id, NotificationSerializer(notification).data)
    return notification


def emit_notification_to_admins(*, type_, body, actor='', case=None):
    """Deliver to every active admin account."""
    from django.contrib.auth import get_user_model

    User = get_user_model()
    for admin in User.objects.filter(role=User.Role.ADMIN, is_active=True):
        emit_notification(recipient=admin, type_=type_, body=body, actor=actor, case=case)


def emit_notification_to_doctors(*, type_, body, actor='', case=None):
    """Deliver to every active doctor. Used to alert the queue of a new case."""
    from django.contrib.auth import get_user_model

    User = get_user_model()
    for doctor in User.objects.filter(role=User.Role.DOCTOR, is_active=True):
        emit_notification(recipient=doctor, type_=type_, body=body, actor=actor, case=case)
