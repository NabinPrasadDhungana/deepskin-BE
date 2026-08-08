"""
Unit tests for the ML pipeline's per-image inference + worst-case roll-up.

These deliberately mock `predict` and `generate_attention_map_overlay` so
the suite never loads TensorFlow or the trained model -- the point here is
to verify the orchestration (loop every image, persist a per-Image result,
roll the highest malignant score up to the Case), not the model itself.
"""
from io import BytesIO
from unittest.mock import patch
import base64

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image

from cases.models import Case, CaseImage
from mlservice import pipeline

User = get_user_model()


def png_bytes() -> bytes:
    buf = BytesIO()
    Image.new('RGB', (10, 10), color='pink').save(buf, format='PNG')
    return buf.getvalue()


def add_image(case, is_primary):
    return CaseImage.objects.create(
        case=case,
        is_primary=is_primary,
        image=SimpleUploadedFile('lesion.png', png_bytes(), content_type='image/png'),
    )


class PipelineInferenceTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='p1', password='pw12345!', role=User.Role.PATIENT
        )
        self.case = Case.objects.create(patient=self.user, patient_note='x')

    @patch('mlservice.pipeline.generate_attention_map_overlay',
           return_value=png_bytes())
    @patch('mlservice.pipeline.predict', side_effect=[0.2, 0.9, 0.5])
    def test_per_image_inference_and_worst_case_rollup(self, mock_predict, mock_attn):
        add_image(self.case, is_primary=True)
        add_image(self.case, is_primary=False)
        add_image(self.case, is_primary=False)

        pipeline.run_inference_on_case(self.case)

        # Every image gets scored + an attention map, once each.
        self.assertEqual(mock_predict.call_count, 3)
        self.assertEqual(mock_attn.call_count, 3)

        ordered = list(self.case.images.all().order_by('id'))
        expected_confs = [0.2, 0.9, 0.5]
        # DEEPSKIN_CLASSIFICATION_THRESHOLD = 0.30
        expected_predictions = ['benign', 'malignant', 'malignant']
        for img, conf, pred in zip(ordered, expected_confs, expected_predictions):
            self.assertEqual(img.ai_confidence, conf)
            self.assertEqual(img.ai_prediction, pred)
            self.assertIsNotNone(img.ai_attention_map)
            self.assertIsNotNone(img.ai_processed_at)

        # Worst-case (highest malignant score) image is rolled up to the Case.
        self.assertEqual(self.case.ai_confidence, 0.9)
        self.assertEqual(self.case.ai_prediction, Case.AIPrediction.MALIGNANT)
        self.assertEqual(self.case.ai_priority, Case.Priority.HIGH)  # 0.9 >= 0.60
        self.assertIsNotNone(self.case.attention_map_image)
        self.assertIsNotNone(self.case.ai_processed_at)

    def test_raises_when_no_images(self):
        with self.assertRaises(ValueError):
            pipeline.run_inference_on_case(self.case)

    @patch('mlservice.pipeline.generate_attention_map_overlay',
           return_value=png_bytes())
    @patch('mlservice.pipeline.predict', side_effect=[0.1, 0.29])
    def test_rollup_benign_when_all_below_threshold(self, mock_predict, mock_attn):
        add_image(self.case, is_primary=True)
        add_image(self.case, is_primary=False)
        pipeline.run_inference_on_case(self.case)
        self.assertEqual(self.case.ai_prediction, Case.AIPrediction.BENIGN)
        self.assertEqual(self.case.ai_priority, Case.Priority.LOW)  # 0.29 < 0.30


class ResultWebhookTestCase(TestCase):
    """The HF Space callback path: token-gated, idempotent, and rolls the
    worst image up to the Case once every image has been scored."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='webhook_patient', password='pw12345!',
            role=User.Role.PATIENT,
        )
        self.case = Case.objects.create(patient=self.user, patient_note='x')
        self.url = reverse('ml-result-webhook')
        self.token = 'test-token'
        self.img = add_image(self.case, is_primary=True)

    def _post(self, confidence=0.9, expect_image_id=None, token=None):
        return self.client.post(
            self.url,
            data={
                'client_ref': 'ref-1',
                'case_id': str(self.case.pk),
                'image_id': expect_image_id or self.img.pk,
                'confidence': confidence,
                'prediction': 'malignant' if confidence >= 0.30 else 'benign',
                'attention_png_base64': base64.b64encode(png_bytes()).decode(),
            },
            format='json',
            HTTP_X_HF_TOKEN=token if token is not None else self.token,
        )

    @override_settings(DEEPSKIN_HF_TOKEN='test-token')
    def test_rejects_missing_or_wrong_token(self):
        self.assertEqual(self._post(token='').status_code, 403)
        self.assertEqual(self._post(token='nope').status_code, 403)

    @override_settings(DEEPSKIN_HF_TOKEN='test-token')
    def test_result_writes_image_and_completes_case(self):
        resp = self._post(confidence=0.9)
        self.assertEqual(resp.status_code, 200)

        self.img.refresh_from_db()
        self.assertEqual(self.img.ai_confidence, 0.9)
        self.assertEqual(self.img.ai_prediction, Case.AIPrediction.MALIGNANT)
        self.assertIsNotNone(self.img.ai_attention_map)

        self.case.refresh_from_db()
        self.assertEqual(self.case.ai_status, Case.AIStatus.DONE)
        self.assertEqual(self.case.ai_confidence, 0.9)

    @override_settings(DEEPSKIN_HF_TOKEN='test-token')
    def test_case_stays_processing_until_all_images_land(self):
        second = add_image(self.case, is_primary=False)

        self._post(confidence=0.9)  # only the first image scored
        self.case.refresh_from_db()
        self.assertEqual(self.case.ai_status, Case.AIStatus.QUEUED)  # still waiting
        # (the task owns the PROCESSING transition; the webhook alone only
        # completes the case once EVERY image has a result)

        self._post(confidence=0.5, expect_image_id=second.pk)
        self.case.refresh_from_db()
        self.assertEqual(self.case.ai_status, Case.AIStatus.DONE)
        self.assertEqual(self.case.ai_confidence, 0.9)  # worst image wins