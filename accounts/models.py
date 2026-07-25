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

    role = models.CharField(max_length=10, choices=Role.choices)

    # Optional clinical fields -- only meaningful for doctors, left blank
    # otherwise. Kept on the base User rather than a separate profile model
    # to avoid over-engineering for this project's scope.
    specialty = models.CharField(max_length=100, blank=True)
    phone_number = models.CharField(max_length=20, blank=True)

    def is_patient(self):
        return self.role == self.Role.PATIENT

    def is_doctor(self):
        return self.role == self.Role.DOCTOR

    def is_admin_role(self):
        # Named to avoid clashing with Django's built-in is_staff/is_superuser
        return self.role == self.Role.ADMIN

    def __str__(self):
        return f'{self.username} ({self.role})'
