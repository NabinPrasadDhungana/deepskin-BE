from django.urls import path

from .views import ResultWebhookView

urlpatterns = [
    path('result/', ResultWebhookView.as_view(), name='ml-result-webhook'),
]