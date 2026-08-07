from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """
    DeepSkin's custom user model. Adds a `role` field that drives RBAC
    throughout the system (see cases/permissions.py for how it's enforced).

    Kept deliberately simple: three roles only (patient, doctor, admin),
    matching the system design agreed for this project's scope.
    """

    class Role(models.TextChoices):
        PATIENT = 'patient', 'Patient'
        DOCTOR = 'doctor', 'Doctor'
        ADMIN = 'admin', 'Admin'
        
    class VerificationStatus(models.TextChoices):
        NOT_APPLICABLE = 'n_a', 'Not applicable'
        PENDING = 'pending', 'Pending review'
        APPROVED = 'approved', 'Approved'
        REJECTED = 'rejected', 'Rejected'

    role = models.CharField(max_length=10, choices=Role.choices)

    
    
    
    specialty = models.CharField(max_length=100, blank=True)
    phone_number = models.CharField(max_length=20, blank=True)
    
    
    license_number = models.CharField(max_length=100, blank=True)
    license_document = models.ImageField(upload_to='doctor_licenses/', null=True, blank=True)
    verification_status = models.CharField(
        max_length=10, choices=VerificationStatus.choices,
        default=VerificationStatus.NOT_APPLICABLE,
    )
    verification_notes = models.TextField(blank=True)
    reviewed_by = models.ForeignKey(
        'self', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='doctor_reviews_done',
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)

    def is_patient(self):
        return self.role == self.Role.PATIENT

    def is_doctor(self):
        return self.role == self.Role.DOCTOR

    def is_admin_role(self):
        
        return self.role == self.Role.ADMIN
    
    def is_verified_doctor(self):
        return self.is_doctor() and self.verification_status == self.VerificationStatus.APPROVED

    def __str__(self):
        return f'{self.username} ({self.role})'
