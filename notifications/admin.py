from django.contrib import admin

from .models import Notification


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ['recipient', 'type', 'actor', 'case', 'read_at', 'created_at']
    list_filter = ['type', 'read_at']
    search_fields = ['recipient__username', 'body']
    readonly_fields = ['created_at']
