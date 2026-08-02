import json

import redis
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.http import StreamingHttpResponse
from drf_spectacular.utils import OpenApiExample, OpenApiParameter, OpenApiResponse, extend_schema
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

    @extend_schema(
        responses={
            200: OpenApiResponse(
                response=NotificationSerializer(many=True),
                description="The current user's notifications, newest first.",
                examples=[
                    OpenApiExample(
                        "Notification list",
                        value=[
                            {
                                "id": 49,
                                "type": "case_submitted",
                                "body": "A new case has been submitted and is now in the queue.",
                                "actor": "e2e_docX",
                                "case": "4c69e38a-1234-4abc-9def-0123456789ab",
                                "created_at": "2026-08-02T17:28:36.898644Z",
                                "read_at": None,
                            }
                        ],
                    )
                ],
            )
        }
    )
    def get(self, request):
        notifications = Notification.objects.filter(recipient=request.user)
        return Response(
            NotificationSerializer(notifications, many=True).data
        )


class MarkNotificationReadView(APIView):
    """POST /api/notifications/<id>/read/ -- mark a single notification read."""

    @extend_schema(
        request=None,
        parameters=[
            OpenApiParameter(
                "id", type=int, location=OpenApiParameter.PATH, required=True,
                examples=[OpenApiExample("id", value=3)],
            )
        ],
        responses={204: None},
    )
    def post(self, request, pk):
        Notification.objects.filter(pk=pk, recipient=request.user).update(read_at=timezone.now())
        return Response(status=204)


class MarkAllNotificationsReadView(APIView):
    """POST /api/notifications/read-all/ -- mark every unread notification read."""

    @extend_schema(request=None, responses={204: None})
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


class NotificationStreamView(APIView):
    """
    GET /api/notifications/stream/

    Server-Sent Events stream of live notifications for the authenticated
    user. Subscribes to the Redis channel ``notify:<user_id>`` and yields
    one ``data:`` line per notification, with a ``: ping`` heartbeat to
    keep the connection and proxies alive.

    Authentication: ``Authorization: Bearer <jwt>`` header (preferred) or
    ``?token=<jwt>`` query param (fallback).  Using a header avoids
    browser/privacy-extension blocking of long tokens in URLs.

    DRF authentication/permission is disabled here because auth is handled
    manually (``_authenticate_stream_request``) to support the ``?token=``
    query-param fallback that ``JWTAuthentication`` does not read.
    """

    authentication_classes = []
    permission_classes = []

    @extend_schema(
        parameters=[
            OpenApiParameter(
                "Authorization", location=OpenApiParameter.HEADER,
                description='Preferred: ``Bearer <jwt>`` (or pass ``?token=<jwt>`` instead).',
                examples=[OpenApiExample("Authorization", value="Bearer <access_token>")],
            )
        ],
        responses={
            200: OpenApiResponse(
                response={
                    "type": "string",
                    "description": (
                        "Server-Sent Events stream. A ``: ping`` heartbeat is "
                        "emitted while idle; each notification arrives as a "
                        "``data:`` line carrying the same payload as the list "
                        "endpoint. Content-Type is actually text/event-stream."
                    ),
                },
                description="Live SSE stream of the current user's notifications.",
                examples=[
                    OpenApiExample(
                        "Event stream",
                        summary="A live notification frame (with preceding heartbeat)",
                        value=(
                            ": ping\n\n"
                            "data: {\"id\": 49, \"type\": \"case_ready\", \"body\": \"AI analysis is complete; the case is ready to pick up.\", \"actor\": \"System\", \"case\": \"4c69e38a-1234-4abc-9def-0123456789ab\", \"created_at\": \"2026-08-02T17:28:36.898644Z\", \"read_at\": null}\n\n"
                        ),
                    )
                ],
            )
        },
    )
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
