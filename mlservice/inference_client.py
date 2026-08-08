"""
Async client for the hosted inference Space (Hugging Face Spaces).

The Django Celery worker does NOT do TF inference itself anymore -- it
submits each case image to HF and gets a 202 immediately. HF runs the
model in a background task and POSTs the result back to the Django
webhook endpoint (see mlservice/views.py ResultWebhookView).

Environment:
    DEEPSKIN_HF_ENDPOINT   e.g. https://<owner>-deepskin-inf.hf.space
    DEEPSKIN_HF_TOKEN      shared secret, also set as a Space secret
"""
import base64
import logging
import uuid

import requests
from django.conf import settings

logger = logging.getLogger('deepskin.hfclient')


class InferenceSubmitError(RuntimeError):
    """The HF Space could not be reached or rejected the submission."""


def submit_image(case_id, image_id, image_bytes) -> dict:
    """
    POST one case image to the HF Space for async inference. Returns
    {client_ref, case_id, image_id} so callers can track what HF should
    echo back in its callback.
    """
    client_ref = str(uuid.uuid4())
    payload = {
        'case_id': str(case_id),
        'image_id': image_id,
        'client_ref': client_ref,
        'image_bytes': base64.b64encode(image_bytes).decode('ascii'),
        'callback_url': settings.DEEPSKIN_HF_CALLBACK_URL,
    }
    headers = {'X-HF-Token': settings.DEEPSKIN_HF_TOKEN}

    url = settings.DEEPSKIN_HF_URL.rstrip('/') + '/predict'
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
    except requests.RequestException as exc:
        raise InferenceSubmitError(f'HF unreachable: {exc}') from exc

    if resp.status_code != 202:
        raise InferenceSubmitError(
            f'HF rejected submission: HTTP {resp.status_code}: {resp.text[:300]}'
        )
    return {'client_ref': client_ref, 'case_id': str(case_id), 'image_id': image_id}