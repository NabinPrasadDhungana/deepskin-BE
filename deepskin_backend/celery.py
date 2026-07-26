"""
Celery app config. ML inference (mlservice.tasks.process_case_task) runs
here instead of inline in the upload request, so patients get an instant
201 response and the doctor queue fills in once processing finishes.
"""
import os

from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepskin_backend.settings')

app = Celery('deepskin_backend')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()