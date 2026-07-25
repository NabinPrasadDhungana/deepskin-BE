"""
DeepSkin model inference pipeline.

Deliberately isolated from `cases/views.py`: the case-workflow code just
calls `run_inference_on_case(case)` and doesn't know or care how the
prediction is produced. This makes it trivial to:
  - swap in the real trained model without touching any view/serializer code
  - mock this function entirely in tests (see mlservice/tests.py) so test
    runs don't need a GPU or the actual 8M-parameter model loaded
  - move this to an async task queue later without changing the call site

The preprocessing steps here (DullRazor hair removal, resize, no manual
normalization) MUST exactly match what was used during model training --
see the project's model development notes. Any mismatch here silently
degrades prediction quality without raising an error, so this is the
single most important place to keep in sync with the training pipeline.
"""
import io

import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone
from PIL import Image

from cases.models import Case

_model = None  # lazy-loaded singleton, see get_model()


def get_model():
    """
    Lazily loads the trained .keras model on first use and caches it.
    Avoids loading an 8M-parameter model (plus TensorFlow's own startup
    cost) on every single request, and avoids loading it at all for
    management commands / tests that never call inference.
    """
    global _model
    if _model is None:
        import tensorflow as tf
        # custom_objects must match the training notebook's CUSTOM_OBJECTS
        # (BinaryFocalLoss, CBAM, ChannelAttention, SpatialAttention) --
        # compile=False since we only need forward-pass inference here.
        from .custom_layers import CUSTOM_OBJECTS
        _model = tf.keras.models.load_model(
            settings.DEEPSKIN_MODEL_PATH,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )
    return _model


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Mirrors the offline preprocessing pipeline used at training time:
    resize to 260x260, keep pixels in [0,255] float32 (EfficientNetB2
    rescales internally -- no manual normalization here).

    NOTE: production DullRazor hair removal is intentionally NOT run here
    inline for latency reasons on first pass; if evaluation shows a material
    accuracy gap between this simplified path and the training pipeline,
    port `dull_razor()` from the preprocessing notebook into this function
    before relying on this for real predictions.
    """
    img = pil_image.convert('RGB').resize((260, 260), Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32)  # [0, 255], matches training
    return arr


def predict(pil_image: Image.Image) -> float:
    """Returns the raw malignant-class probability (0.0-1.0)."""
    model = get_model()
    x = preprocess_image(pil_image)
    x = np.expand_dims(x, axis=0)  # add batch dimension
    prob = float(model.predict(x, verbose=0)[0][0])
    return prob


def confidence_to_priority(confidence: float) -> str:
    """Buckets the raw confidence into High/Medium/Low for the doctor
    queue view -- see settings.py for the cutoff values."""
    if confidence >= settings.DEEPSKIN_PRIORITY_HIGH_CUTOFF:
        return Case.Priority.HIGH
    if confidence >= settings.DEEPSKIN_PRIORITY_MEDIUM_CUTOFF:
        return Case.Priority.MEDIUM
    return Case.Priority.LOW


def generate_attention_map_overlay(pil_image: Image.Image) -> bytes:
    """
    Produces the CBAM spatial-attention heatmap overlaid on the original
    image, for the doctor's case-detail view. See the model development
    notebook's "CBAM Attention Map Visualisation" section for the reference
    implementation this should mirror (att_model built from the model's
    'cbam' layer, cv2.applyColorMap + addWeighted for the overlay).

    Returns PNG bytes ready to save onto Case.attention_map_image.
    Left as a TODO stub wired into the pipeline so the field/flow exists
    end-to-end now; drop in the real heatmap generation once the trained
    model file is deployed alongside this backend.
    """
    # TODO: replace with real CBAM attention extraction (see notebook).
    buf = io.BytesIO()
    pil_image.convert('RGB').resize((260, 260)).save(buf, format='PNG')
    return buf.getvalue()


def run_inference_on_case(case: Case) -> None:
    """
    The single entry point cases/views.py calls after a patient uploads a
    case. Populates ai_confidence, ai_priority, attention_map_image, and
    ai_processed_at on the Case in place.
    """
    primary_image = case.images.filter(is_primary=True).first() or case.images.first()
    if primary_image is None:
        return

    pil_image = Image.open(primary_image.image)
    confidence = predict(pil_image)

    case.ai_confidence = confidence
    case.ai_priority = confidence_to_priority(confidence)

    overlay_bytes = generate_attention_map_overlay(pil_image)
    case.attention_map_image.save(
        f'{case.id}_attention.png', ContentFile(overlay_bytes), save=False
    )

    case.ai_processed_at = timezone.now()
    case.save(update_fields=[
        'ai_confidence', 'ai_priority', 'attention_map_image', 'ai_processed_at'
    ])
