from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.request import Request
from rest_framework.test import APIRequestFactory, APITestCase
from rest_framework_simplejwt.tokens import RefreshToken

from cases.models import Case
from mlservice.tasks import process_case_task
from notifications.models import Notification
from notifications.views import NotificationStreamView, _authenticate_stream_request

User = get_user_model()


@override_settings(DEEPSKIN_HF_TOKEN='test-hf-token', CELERY_TASK_ALWAYS_EAGER=True)
class NotificationTests(APITestCase):
    def setUp(self):
        self.patient = User.objects.create_user(
            username='notif_patient', password='pw12345!', role=User.Role.PATIENT
        )
        self.doctor = User.objects.create_user(
            username='notif_doctor', password='pw12345!', role=User.Role.DOCTOR
        )
        self.admin = User.objects.create_user(
            username='notif_admin', password='pw12345!', role=User.Role.ADMIN
        )
        self.case = Case.objects.create(
            patient=self.patient,
            patient_note='notification test',
            ai_status=Case.AIStatus.DONE,
        )

    def auth(self, user):
        self.client.force_authenticate(user=user)

    def _png_file(self):
        from io import BytesIO

        from django.core.files.uploadedfile import SimpleUploadedFile
        from PIL import Image

        buf = BytesIO()
        Image.new('RGB', (10, 10), color='pink').save(buf, format='PNG')
        return SimpleUploadedFile('lesion.png', buf.getvalue(), content_type='image/png')

    def _deliver_result(self, confidence=0.82):
        """POST a fake HF inference callback for the case's (single) image,
        mirroring the production result webhook flow."""
        img = self.case.images.first()
        resp = self.client.post(
            reverse('ml-result-webhook'),
            data={
                'client_ref': 'test-ref',
                'case_id': str(self.case.pk),
                'image_id': img.pk,
                'confidence': confidence,
                'prediction': 'malignant' if confidence >= 0.30 else 'benign',
            },
            format='json',
            HTTP_X_HF_TOKEN='test-hf-token',
        )
        self.assertEqual(resp.status_code, 200, resp.content)

    def test_notifications_created_at_transitions(self):
        # Doctor picks up, messages the patient, then submits a verdict.
        self.auth(self.doctor)
        self.assertEqual(self.client.post(reverse('case-pickup', args=[self.case.pk])).status_code, 200)
        self.assertEqual(
            self.client.post(reverse('case-messages', args=[self.case.pk]), {'body': 'Hello!'}).status_code, 201
        )
        self.assertEqual(
            self.client.post(reverse('case-verdict', args=[self.case.pk]), {'decision': 'monitor', 'notes': 'OK'}).status_code,
            201,
        )

        self.auth(self.patient)
        resp = self.client.get(reverse('notification-list'))
        self.assertEqual(resp.status_code, 200)
        types = [n['type'] for n in resp.data]
        self.assertIn('pickup', types)
        self.assertIn('message', types)
        self.assertIn('verdict', types)
        # Patient notification bodies must never leak AI output.
        for n in resp.data:
            self.assertNotIn('confidence', n['body'])
            self.assertNotIn('prediction', n['body'])
            self.assertNotIn('attention', n['body'])

    def test_message_notifies_doctor_when_patient_replies(self):
        self.case.assigned_doctor = self.doctor
        self.case.status = Case.Status.IN_REVIEW
        self.case.save()

        self.auth(self.patient)
        self.assertEqual(
            self.client.post(reverse('case-messages', args=[self.case.pk]), {'body': 'Thank you'}).status_code, 201
        )

        self.auth(self.doctor)
        data = self.client.get(reverse('notification-list')).data
        self.assertEqual([n['type'] for n in data], ['message'])

    def test_mark_read_endpoints(self):
        Notification.objects.create(recipient=self.patient, type='message', body='a')
        Notification.objects.create(recipient=self.patient, type='verdict', body='b')

        self.auth(self.patient)
        first = Notification.objects.filter(recipient=self.patient).first()
        self.assertEqual(self.client.post(reverse('notification-read', args=[first.pk])).status_code, 204)

        data = self.client.get(reverse('notification-list')).data
        single = next(n for n in data if n['id'] == first.pk)
        self.assertIsNotNone(single['read_at'])

        self.assertEqual(self.client.post(reverse('notification-read-all')).status_code, 204)
        data = self.client.get(reverse('notification-list')).data
        self.assertTrue(all(n['read_at'] for n in data))

    @patch('mlservice.tasks.submit_image', return_value={})
    def test_ai_done_notifies_patient(self, submit):
        # Task hands the image to HF; the CASE_READY push is emitted by the
        # result webhook once processing completes (mirrors production).
        self.case.images.create(image=self._png_file(), is_primary=True)
        process_case_task.apply(args=[str(self.case.pk)])
        submit.assert_called_once()

        # Simulate the HF callback that completes the case.
        self._deliver_result(0.82)
        self.case.refresh_from_db()
        self.assertEqual(self.case.ai_status, Case.AIStatus.DONE)
        notif = Notification.objects.get(recipient=self.patient)
        self.assertEqual(notif.type, Notification.Type.CASE_READY)

    @patch('mlservice.tasks.submit_image', side_effect=RuntimeError('boom'))
    def test_ai_failed_notifies_admins(self, submit):
        self.case.images.create(image=self._png_file(), is_primary=True)
        with self.assertRaises(Exception):
            process_case_task.apply(args=[str(self.case.pk)]).get()
        # Each retry re-emits, so assert at least one admin alert was created.
        self.assertGreaterEqual(
            Notification.objects.filter(recipient=self.admin, type=Notification.Type.AI_FAILED).count(), 1
        )

    def test_case_submitted_notifies_doctors_and_admins(self):
        """Case submission (CASE_SUBMITTED) alerts every active doctor + admin."""
        from io import BytesIO
        from unittest.mock import patch

        from PIL import Image

        buf = BytesIO()
        Image.new('RGB', (10, 10)).save(buf, format='JPEG')
        buf.seek(0)
        buf.name = 'lesion.jpg'

        self.auth(self.patient)
        with patch('mlservice.pipeline.run_inference_on_case', return_value=None):
            resp = self.client.post(
                reverse('case-list-create'),
                {'patient_note': 'new lesion', 'images': [buf]},
                format='multipart',
            )
        self.assertEqual(resp.status_code, 201, resp.content)
        case_id = resp.data['id']

        doctor_case_submitted = Notification.objects.filter(
            recipient=self.doctor, type='case_submitted', case_id=case_id
        )
        admin_case_submitted = Notification.objects.filter(
            recipient=self.admin, type='case_submitted', case_id=case_id
        )
        self.assertTrue(doctor_case_submitted.exists())
        self.assertTrue(admin_case_submitted.exists())
        self.assertNotIn('confidence', doctor_case_submitted.first().body)

    def test_doctor_application_notifies_admins(self):
        from io import BytesIO
        from PIL import Image

        buf = BytesIO()
        Image.new('RGB', (10, 10)).save(buf, format='JPEG')
        buf.seek(0)
        buf.name = 'license.jpg'

        resp = self.client.post(
            reverse('doctor-self-register'),
            {
                'username': 'new_doctor', 'email': 'nd@x.com', 'password': 'pw12345!',
                'specialty': 'Dermatology', 'license_number': 'MED-1',
                'license_document': buf,
            },
            format='multipart',
        )
        self.assertEqual(resp.status_code, 201, resp.content)
        notif = Notification.objects.filter(recipient=self.admin, type=Notification.Type.APPLICATION).first()
        self.assertIsNotNone(notif)
        self.assertIn('new_doctor', notif.body)

    def test_stream_auth(self):
        token = str(RefreshToken.for_user(self.patient).access_token)
        factory = APIRequestFactory()

        # via Authorization header (preferred)
        ok_request = Request(factory.get('/api/notifications/stream/', HTTP_AUTHORIZATION=f'Bearer {token}'))
        self.assertEqual(_authenticate_stream_request(ok_request), self.patient)

        # via query param (fallback)
        ok_request_q = Request(factory.get('/api/notifications/stream/', {'token': token}))
        self.assertEqual(_authenticate_stream_request(ok_request_q), self.patient)

        bad_request = Request(factory.get('/api/notifications/stream/', {'token': 'garbage'}))
        with self.assertRaises(AuthenticationFailed):
            _authenticate_stream_request(bad_request)

        missing_request = Request(factory.get('/api/notifications/stream/'))
        with self.assertRaises(AuthenticationFailed):
            _authenticate_stream_request(missing_request)

    def test_stream_response_headers(self):
        token = str(RefreshToken.for_user(self.patient).access_token)
        factory = APIRequestFactory()
        request = Request(factory.get('/api/notifications/stream/', {'token': token}))
        response = NotificationStreamView().get(request)
        self.assertEqual(response['Content-Type'], 'text/event-stream')
        self.assertEqual(response['Cache-Control'], 'no-cache')
