"""
End-to-end workflow test using Django's APIClient. Exercises the full
patient -> AI -> doctor queue -> pickup -> verdict -> patient view path,
and specifically asserts the RBAC boundary that a patient's case detail
response NEVER contains the AI confidence/priority/attention-map fields --
this is the single most safety-critical behaviour in the whole system.

The suite runs with DEEPSKIN_CELERY_EAGER=True, so uploads schedule
process_case_task in-process. Since inference now runs on the HF Space
(not locally), this test patches mlservice.tasks.submit_image (the HTTP
handoff) and then drives the result webhook endpoint with fabricated
results -- the exact path the real HF Space uses.
"""
from io import BytesIO
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse
from PIL import Image
from rest_framework.test import APITestCase

from .models import Case

User = get_user_model()

HF_TOKEN = 'test-hf-token'


def fake_image_file():
    buf = BytesIO()
    Image.new('RGB', (50, 50), color='pink').save(buf, format='JPEG')
    buf.seek(0)
    buf.name = 'lesion.jpg'
    return buf


def deliver_ai_result(client, case, confidences):
    """Simulate HF callbacks: one POST to the result webhook per image."""
    for img, conf in zip(case.images.all(), confidences):
        resp = client.post(
            reverse('ml-result-webhook'),
            data={
                'client_ref': 'test-ref',
                'case_id': str(case.id),
                'image_id': img.id,
                'confidence': conf,
                'prediction': (
                    Case.AIPrediction.MALIGNANT if conf >= 0.30 else Case.AIPrediction.BENIGN
                ),
            },
            format='json',
            HTTP_X_HF_TOKEN=HF_TOKEN,
        )
        assert resp.status_code == 200, resp.content


@override_settings(
    DEEPSKIN_HF_TOKEN=HF_TOKEN,
    DEEPSKIN_HF_URL='http://testserver',
    CELERY_TASK_ALWAYS_EAGER=True,
)
class DeepSkinWorkflowTests(APITestCase):
    def setUp(self):
        self.patient = User.objects.create_user(
            username='patient1', password='pw12345!', role=User.Role.PATIENT
        )
        self.doctor = User.objects.create_user(
            username='doctor1', password='pw12345!', role=User.Role.DOCTOR
        )

    def auth(self, user):
        self.client.force_authenticate(user=user)

    @patch('mlservice.tasks.submit_image', return_value={})
    def test_full_workflow_and_patient_never_sees_ai_fields(self, mock_submit):
        # 1. Patient uploads a case; async AI handoff is mocked
        self.auth(self.patient)
        resp = self.client.post(
            reverse('case-list-create'),
            {'patient_note': 'itchy mole', 'images': [fake_image_file()]},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 201, resp.content)
        case_id = resp.data['id']
        mock_submit.assert_called_once()

        # Deliver AI results through the webhook path (as HF would).
        case = Case.objects.get(pk=case_id)
        deliver_ai_result(self.client, case, [0.82])

        # Patient detail view must NOT contain any AI fields whatsoever
        detail = self.client.get(reverse('case-detail-patient', args=[case_id]))
        self.assertEqual(detail.status_code, 200)
        for forbidden_field in ('ai_confidence', 'ai_priority', 'attention_map_image'):
            self.assertNotIn(forbidden_field, detail.data)

        # 2. Doctor sees it in the queue (coarse priority only)
        self.auth(self.doctor)
        queue = self.client.get(reverse('case-queue'))
        self.assertEqual(queue.status_code, 200)
        queue_ids = [c['id'] for c in queue.data['results']]
        self.assertIn(case_id, queue_ids)
        queue_item = next(c for c in queue.data['results'] if c['id'] == case_id)
        self.assertNotIn('ai_confidence', queue_item)  # bucket only, not raw score

        # 3. Doctor picks it up
        pickup = self.client.post(reverse('case-pickup', args=[case_id]))
        self.assertEqual(pickup.status_code, 200, pickup.content)
        self.assertEqual(pickup.data['ai_confidence'], 0.82)  # full detail now visible

        # A second doctor may not also pick it up
        other_doctor = User.objects.create_user(
            username='doctor2', password='pw12345!', role=User.Role.DOCTOR
        )
        self.auth(other_doctor)
        second_pickup = self.client.post(reverse('case-pickup', args=[case_id]))
        self.assertEqual(second_pickup.status_code, 409)

        # 4. Original doctor records a verdict
        self.auth(self.doctor)
        verdict = self.client.post(
            reverse('case-verdict', args=[case_id]),
            {'decision': 'biopsy', 'notes': 'Irregular border, recommend biopsy.'},
        )
        self.assertEqual(verdict.status_code, 201, verdict.content)

        case = Case.objects.get(pk=case_id)
        self.assertEqual(case.status, Case.Status.REVIEWED)

        # 5. Patient now sees the verdict but still no AI fields
        self.auth(self.patient)
        final_detail = self.client.get(reverse('case-detail-patient', args=[case_id]))
        self.assertEqual(final_detail.data['status'], 'reviewed')
        self.assertEqual(final_detail.data['verdict']['decision'], 'biopsy')
        for forbidden_field in ('ai_confidence', 'ai_priority', 'attention_map_image'):
            self.assertNotIn(forbidden_field, final_detail.data)

    @patch('mlservice.tasks.submit_image', return_value={})
    def test_messaging_between_patient_and_assigned_doctor(self, mock_submit):
        self.auth(self.patient)
        resp = self.client.post(
            reverse('case-list-create'),
            {'patient_note': 'concerned', 'images': [fake_image_file()]},
            format='multipart',
        )
        case_id = resp.data['id']

        self.auth(self.doctor)
        self.client.post(reverse('case-pickup', args=[case_id]))

        # Doctor messages patient
        msg = self.client.post(
            reverse('case-messages', args=[case_id]),
            {'body': 'Can you tell me how long this has been there?'},
        )
        self.assertEqual(msg.status_code, 201, msg.content)

        # Patient can see and reply
        self.auth(self.patient)
        thread = self.client.get(reverse('case-messages', args=[case_id]))
        self.assertEqual(len(thread.data['results']), 1)

        reply = self.client.post(
            reverse('case-messages', args=[case_id]),
            {'body': 'About 3 weeks.'},
        )
        self.assertEqual(reply.status_code, 201)

        # A stranger doctor may not read this thread
        stranger = User.objects.create_user(
            username='doctor3', password='pw12345!', role=User.Role.DOCTOR
        )
        self.auth(stranger)
        blocked = self.client.get(reverse('case-messages', args=[case_id]))
        self.assertEqual(len(blocked.data['results']), 0)

    @patch('mlservice.tasks.submit_image', return_value={})
    def test_per_image_ai_fields_exposed_to_doctor_only(self, mock_submit):
        """Upload two images; confirm both get per-image AI results, that a
        doctor sees them on each image, and that a patient never does -- the
        per-image safety boundary."""
        self.auth(self.patient)
        resp = self.client.post(
            reverse('case-list-create'),
            {'patient_note': 'two angles', 'images': [fake_image_file(), fake_image_file()]},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 201, resp.content)
        case_id = resp.data['id']
        case = Case.objects.get(pk=case_id)
        self.assertEqual(case.images.count(), 2)

        deliver_ai_result(self.client, case, [0.82, 0.82])

        # Patient detail -- neither case-level nor per-image AI fields.
        pdetail = self.client.get(reverse('case-detail-patient', args=[case_id]))
        self.assertEqual(pdetail.status_code, 200)
        for forbidden in ('ai_confidence', 'ai_priority', 'attention_map_image'):
            self.assertNotIn(forbidden, pdetail.data)
        self.assertEqual(len(pdetail.data['images']), 2)
        for img in pdetail.data['images']:
            for forbidden in ('ai_confidence', 'ai_prediction', 'ai_attention_map'):
                self.assertNotIn(forbidden, img)

        # Doctor: per-image AI fields present on every image.
        self.auth(self.doctor)
        self.client.post(reverse('case-pickup', args=[case_id]))
        doctor_detail = self.client.get(reverse('case-detail-doctor', args=[case_id]))
        self.assertEqual(doctor_detail.status_code, 200)
        imgs = doctor_detail.data['images']
        self.assertEqual(len(imgs), 2)
        for img in imgs:
            self.assertEqual(img['ai_prediction'], 'malignant')
            self.assertEqual(img['ai_confidence'], 0.82)
            self.assertIn('ai_attention_map', img)
        # Case-level roll-up is still exposed to the doctor.
        self.assertEqual(doctor_detail.data['ai_prediction'], 'malignant')
        self.assertEqual(doctor_detail.data['ai_confidence'], 0.82)