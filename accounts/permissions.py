"""
Shared role-based permission classes.

Kept in `accounts` (not duplicated per-app) since role checks are the same
everywhere: they just look at `request.user.role`. Each app then combines
these with its own object-level rules (e.g. "a doctor may only see cases
they picked up") where needed -- see cases/permissions.py for those.
"""
from rest_framework.permissions import BasePermission


class IsPatient(BasePermission):
    message = 'This action is only available to patients.'

    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and request.user.is_patient())


class IsDoctor(BasePermission):
    message = 'This action is only available to doctors.'

    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and request.user.is_doctor())


class IsAdminRole(BasePermission):
    message = 'This action is only available to administrators.'

    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and request.user.is_admin_role())


class IsDoctorOrAdmin(BasePermission):
    message = 'This action is only available to doctors or administrators.'

    def has_permission(self, request, view):
        return bool(
            request.user and request.user.is_authenticated
            and (request.user.is_doctor() or request.user.is_admin_role())
        )
