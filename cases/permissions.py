"""
Object-level permission checks for cases. Role checks (is this a doctor at
all?) live in accounts/permissions.py; this file is specifically about
"is THIS user allowed to touch THIS case" -- e.g. a doctor should only see
full detail on cases they've personally picked up, and a patient should
only ever see their own cases.
"""
from rest_framework.permissions import BasePermission, SAFE_METHODS


class IsCaseOwnerPatient(BasePermission):
    """Patient may only access their own cases."""
    message = "You may only access your own cases."

    def has_object_permission(self, request, view, obj):
        return request.user.is_patient() and obj.patient_id == request.user.id


class IsAssignedDoctorOrUnassigned(BasePermission):
    """
    A doctor may view/act on a case if:
      - it's unassigned (queue browsing / pickup), or
      - they are the doctor already assigned to it.
    They may NOT reach into a case another doctor has already picked up.
    """
    message = "This case is assigned to another doctor."

    def has_object_permission(self, request, view, obj):
        if not request.user.is_doctor():
            return False
        if obj.assigned_doctor_id is None:
            return True
        return obj.assigned_doctor_id == request.user.id


class IsAdminOrReadOnlyForOwner(BasePermission):
    """Generic fallback: admins do anything; everyone else read-only on
    objects they don't otherwise own (used sparingly, kept for completeness)."""

    def has_object_permission(self, request, view, obj):
        if request.user.is_admin_role():
            return True
        return request.method in SAFE_METHODS
