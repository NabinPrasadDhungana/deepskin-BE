from django.urls import path

from .views import (
    AdminCaseAuditListView,
    AdminStatsView,
    CaseMessageListCreateView,
    DoctorCaseDetailView,
    DoctorMyCasesView,
    DoctorQueueView,
    PatientCaseDetailView,
    PatientCaseListCreateView,
    PatientHistoryForDoctorView,
    PickUpCaseView,
    RecordVerdictView,
)

urlpatterns = [
    
    path('mine/', PatientCaseListCreateView.as_view(), name='case-list-create'),
    path('mine/<uuid:pk>/', PatientCaseDetailView.as_view(), name='case-detail-patient'),

    
    path('queue/', DoctorQueueView.as_view(), name='case-queue'),
    path('mine-as-doctor/', DoctorMyCasesView.as_view(), name='case-list-doctor'),
    path('<uuid:pk>/detail/', DoctorCaseDetailView.as_view(), name='case-detail-doctor'),
    path('<uuid:pk>/pickup/', PickUpCaseView.as_view(), name='case-pickup'),
    path('<uuid:pk>/verdict/', RecordVerdictView.as_view(), name='case-verdict'),
    path('patient-history/<int:patient_id>/', PatientHistoryForDoctorView.as_view(), name='patient-history'),

    
    path('<uuid:pk>/messages/', CaseMessageListCreateView.as_view(), name='case-messages'),

    
    path('admin/audit/', AdminCaseAuditListView.as_view(), name='admin-audit'),
    path('admin/stats/', AdminStatsView.as_view(), name='admin-stats'),
]
