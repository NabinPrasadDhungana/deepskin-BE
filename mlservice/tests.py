"""
Unit tests for the ML pipeline's per-image inference + worst-case roll-up.

These deliberately mock `predict` and `generate_attention_map_overlay` so
the suite never loads TensorFlow or the trained model -- the point here is
to verify the orchestration (loop every image, persist a per-Image result,
roll the highest malignant score up to the Case), not the model itself.
"""
from io import BytesIO
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
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

        
        self.assertEqual(mock_predict.call_count, 3)
        self.assertEqual(mock_attn.call_count, 3)

        ordered = list(self.case.images.all().order_by('id'))
        expected_confs = [0.2, 0.9, 0.5]
        
        expected_predictions = ['benign', 'malignant', 'malignant']
        for img, conf, pred in zip(ordered, expected_confs, expected_predictions):
            self.assertEqual(img.ai_confidence, conf)
            self.assertEqual(img.ai_prediction, pred)
            self.assertIsNotNone(img.ai_attention_map)
            self.assertIsNotNone(img.ai_processed_at)

        
        self.assertEqual(self.case.ai_confidence, 0.9)
        self.assertEqual(self.case.ai_prediction, Case.AIPrediction.MALIGNANT)
        self.assertEqual(self.case.ai_priority, Case.Priority.HIGH)
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
        self.assertEqual(self.case.ai_priority, Case.Priority.LOW)