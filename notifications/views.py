import json

import redis
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.http import StreamingHttpResponse
from django.views import View
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework_simplejwt.exceptions import TokenError

from .models import Notification
from .serializers import NotificationSerializer

User = get_user_model()

HEARTBEAT_SECONDS = 20


class NotificationListView(APIView):
    """GET /api/notifications/ -- the current user's notifications, newest first."""

    def get(self, request):
        notifications = Notification.objects.filter(recipient=request.user)
        return Response(
            NotificationSerializer(notifications, many=True).data
        )


class MarkNotificationReadView(APIView):
    """POST /api/notifications/<id>/read/ -- mark a single notification read."""

    def post(self, request, pk):
        Notification.objects.filter(pk=pk, recipient=request.user).update(read_at=timezone.now())
        return Response(status=204)


class MarkAllNotificationsReadView(APIView):
    """POST /api/notifications/read-all/ -- mark every unread notification read."""

    def post(self, request):
        Notification.objects.filter(recipient=request.user, read_at__isnull=True).update(
            read_at=timezone.now()
        )
        return Response(status=204)


def _authenticate_stream_request(request):
    """
    Accept the JWT via the ``Authorization`` header (preferred) or the
    ``token`` query param (legacy / fallback).  Returns the user or
    raises AuthenticationFailed.
    """
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
    if auth_header.startswith('Bearer '):
        raw = auth_header[7:]
    else:
        raw = request.GET.get('token')
    if not raw:
        raise AuthenticationFailed('Missing token.')
    try:
        payload = AccessToken(raw)
    except (TokenError, TypeError) as exc:
        raise AuthenticationFailed('Invalid or expired token.') from exc
    try:
        user = User.objects.get(pk=payload['user_id'], is_active=True)
    except User.DoesNotExist as exc:
        raise AuthenticationFailed('User does not exist.') from exc
    return user


class NotificationStreamView(View):
    """
    GET /api/notifications/stream/

    Server-Sent Events stream of live notifications for the authenticated
    user. Subscribes to the Redis channel ``notify:<user_id>`` and yields
    one ``data:`` line per notification, with a ``: ping`` heartbeat to
    keep the connection and proxies alive.

    Authentication: ``Authorization: Bearer <jwt>`` header (preferred) or
    ``?token=<jwt>`` query param (fallback).  Using a header avoids
    browser/privacy-extension blocking of long tokens in URLs.
    """

    def get(self, request):
        try:
            user = _authenticate_stream_request(request)
        except AuthenticationFailed:
            return StreamingHttpResponse(status=401)

        def event_stream(user_id):
            r = redis.Redis.from_url(settings.CELERY_BROKER_URL)
            pubsub = r.pubsub()
            pubsub.subscribe(f'notify:{user_id}')
            try:
                while True:
                    message = pubsub.get_message(timeout=HEARTBEAT_SECONDS)
                    if message and message.get('type') == 'message':
                        yield f"data: {message['data'].decode('utf-8')}\n\n"
                    else:
                        yield ': ping\n\n'
            finally:
                pubsub.unsubscribe(f'notify:{user_id}')
                pubsub.close()
                r.close()

        response = StreamingHttpResponse(
            event_stream(user.id),
            content_type='text/event-stream',
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response
