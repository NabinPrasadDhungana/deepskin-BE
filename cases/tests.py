"""
End-to-end workflow test using Django's APIClient. Exercises the full
patient -> AI -> doctor queue -> pickup -> verdict -> patient view path,
and specifically asserts the RBAC boundary that a patient's case detail
response NEVER contains the AI confidence/priority/attention-map fields --
this is the single most safety-critical behaviour in the whole system.
"""
from io import BytesIO
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone
from PIL import Image
from rest_framework.test import APITestCase

from .models import Case

User = get_user_model()


def fake_image_file():
    buf = BytesIO()
    Image.new('RGB', (50, 50), color='pink').save(buf, format='JPEG')
    buf.seek(0)
    buf.name = 'lesion.jpg'
    return buf


def fake_inference(case):
    """Stands in for mlservice.pipeline.run_inference_on_case -- see
    mlservice/tests.py for why we don't hit the real model here. Runs
    synchronously because the suite sets DEEPSKIN_CELERY_EAGER=True (see
    settings.py) -- process_case_task then executes in-process and calls
    this instead of the real pipeline. Mirrors the real per-image behaviour:
    every CaseImage gets a score, and the Case carries the worst-case roll-up."""
    for img in case.images.all():
        img.ai_confidence = 0.82
        img.ai_prediction = Case.AIPrediction.MALIGNANT
        img.ai_processed_at = timezone.now()
        img.save(update_fields=['ai_confidence', 'ai_prediction', 'ai_processed_at'])

    case.ai_confidence = 0.82
    case.ai_prediction = Case.AIPrediction.MALIGNANT
    case.ai_priority = Case.Priority.HIGH
    case.save(update_fields=['ai_confidence', 'ai_prediction', 'ai_priority'])


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

    @patch('mlservice.pipeline.run_inference_on_case', side_effect=fake_inference)
    def test_full_workflow_and_patient_never_sees_ai_fields(self, mock_infer):
        # 1. Patient uploads a case
        self.auth(self.patient)
        resp = self.client.post(
            reverse('case-list-create'),
            {'patient_note': 'itchy mole', 'images': [fake_image_file()]},
            format='multipart',
        )
        self.assertEqual(resp.status_code, 201, resp.content)
        case_id = resp.data['id']
        mock_infer.assert_called_once()

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

    @patch('mlservice.pipeline.run_inference_on_case', side_effect=fake_inference)
    def test_messaging_between_patient_and_assigned_doctor(self, mock_infer):
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

    @patch('mlservice.pipeline.run_inference_on_case', side_effect=fake_inference)
    def test_per_image_ai_fields_exposed_to_doctor_only(self, mock_infer):
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
