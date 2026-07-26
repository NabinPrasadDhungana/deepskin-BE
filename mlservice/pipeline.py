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
normalization) mirror the offline training pipeline exactly -- see the
model development notebook's preprocessing cells. Any mismatch here
silently degrades prediction quality without raising an error, so this
is the single most important place to keep in sync with training.
"""
import io

import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone
from PIL import Image

from cases.models import Case

_model = None          # lazy-loaded singleton, see get_model()
_attention_model = None  # lazy-built singleton, see get_attention_model()


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
        from .custom_layers import CUSTOM_OBJECTS
        _model = tf.keras.models.load_model(
            settings.DEEPSKIN_MODEL_PATH,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )
    return _model


def get_attention_model():
    """
    Lazily builds a sub-model that outputs the CBAM spatial-attention mask
    instead of the final classification. Mirrors the model development
    notebook's "CBAM Attention Map Visualisation" cell exactly -- built
    from the loaded model's 'cbam' layer, not a separately saved file.
    """
    global _attention_model
    if _attention_model is None:
        import tensorflow as tf

        model = get_model()
        cbam_layer = model.get_layer('cbam')
        cbam_input_tensor = cbam_layer.input

        def get_spatial_mask(x):
            ch_out = cbam_layer.channel_att(x)
            avg = tf.reduce_mean(ch_out, axis=-1, keepdims=True)
            mx = tf.reduce_max(ch_out, axis=-1, keepdims=True)
            combined = tf.concat([avg, mx], axis=-1)
            return cbam_layer.spatial_att.conv(combined)

        _attention_model = tf.keras.Model(
            inputs=model.input,
            outputs=tf.keras.layers.Lambda(get_spatial_mask)(cbam_input_tensor),
            name='attention_map_model',
        )
    return _attention_model


def dull_razor(image: np.ndarray) -> np.ndarray:
    """
    DullRazor algorithm for dermoscopic hair removal (Lee et al., 1997).
    MUST match the training notebook's version exactly -- adaptive kernel
    scaled to ~3.5% of the shorter image dimension, blackhat + TELEA inpaint.

    Args:
        image: RGB uint8 array.
    Returns:
        RGB uint8 array with hair inpainted.
    """
    import cv2

    assert image.dtype == np.uint8, 'Input must be uint8'
    assert image.ndim == 3 and image.shape[2] == 3, 'Input must be RGB'

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    h, w = image.shape[:2]
    k = max(9, int(min(h, w) * 0.035))
    k = k if k % 2 == 1 else k + 1  # ensure odd

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hair_mask = cv2.dilate(hair_mask, dilate_k, iterations=1)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(image_bgr, hair_mask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Full preprocessing pipeline matching training exactly:
    DullRazor -> mild Gaussian blur (BEFORE resize) -> resize to 260x260.
    Returns float32 [0,255] -- EfficientNetB2 rescales internally, no
    manual normalization here.
    """
    import cv2

    img = np.asarray(pil_image.convert('RGB'))  # uint8, original resolution

    img = dull_razor(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (260, 260), interpolation=cv2.INTER_AREA)

    return img.astype(np.float32)  # [0, 255], matches training


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
    import cv2

    att_model = get_attention_model()
    x = preprocess_image(pil_image)
    x_batched = np.expand_dims(x, axis=0)

    mask = att_model.predict(x_batched, verbose=0)[0, :, :, 0]  # tiny grid, e.g. ~9x9

    # FIX: INTER_CUBIC instead of default INTER_LINEAR, plus a Gaussian
    # blur pass -- upsampling a ~9x9 attention grid to 260x260 with plain
    # bilinear interpolation produces visible polygonal facets between
    # grid cells. Cubic interpolation + blur gives a natural-looking
    # heatmap gradient instead, which is what's actually useful to a doctor.
    mask = cv2.resize(mask, (260, 260), interpolation=cv2.INTER_CUBIC)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = np.clip(mask, 0, None)  # cubic interpolation can dip slightly negative
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    base_img = np.clip(x, 0, 255).astype(np.uint8)
    overlay = cv2.addWeighted(base_img, 0.6, heatmap, 0.4, 0)

    success, buf = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError('Failed to encode attention map overlay as PNG.')
    return buf.tobytes()


def run_inference_on_case(case: Case) -> None:
    """
    Called by mlservice.tasks.process_case_task. Populates ai_confidence,
    ai_priority, and attention_map_image on the Case. Status transitions
    (QUEUED -> PROCESSING -> DONE/FAILED) are owned by the calling task,
    not this function -- keeps this function a pure "do the ML work" step.
    """
    primary_image = case.images.filter(is_primary=True).first() or case.images.first()
    if primary_image is None:
        raise ValueError('Case has no images to process.')

    pil_image = Image.open(primary_image.image)
    confidence = predict(pil_image)

    case.ai_confidence = confidence
    case.ai_priority = confidence_to_priority(confidence)

    overlay_bytes = generate_attention_map_overlay(pil_image)
    case.attention_map_image.save(
        f'{case.id}_attention.png', ContentFile(overlay_bytes), save=False
    )

    case.ai_processed_at = timezone.now()
    case.save(update_fields=['ai_confidence', 'ai_priority', 'attention_map_image', 'ai_processed_at'])