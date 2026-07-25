from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import User


@admin.register(User)
class DeepSkinUserAdmin(UserAdmin):
    """Adds `role` to Django's built-in admin so it's visible/editable
    from /admin/ during development without needing a custom dashboard yet."""
    list_display = ('username', 'email', 'role', 'is_active', 'date_joined')
    list_filter = UserAdmin.list_filter + ('role',)
    fieldsets = UserAdmin.fieldsets + (
        ('DeepSkin role', {'fields': ('role', 'specialty', 'phone_number')}),
    )
