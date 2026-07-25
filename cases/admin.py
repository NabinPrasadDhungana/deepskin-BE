from django.contrib import admin

from .models import Case, CaseImage, Message, Verdict


class CaseImageInline(admin.TabularInline):
    model = CaseImage
    extra = 0
    readonly_fields = ('uploaded_at',)


class VerdictInline(admin.StackedInline):
    model = Verdict
    extra = 0


@admin.register(Case)
class CaseAdmin(admin.ModelAdmin):
    """Gives an Admin a working audit-log view immediately via Django admin,
    ahead of the dedicated Admin Audit Log screen being built on the frontend."""
    list_display = ('id', 'patient', 'status', 'ai_priority', 'assigned_doctor', 'created_at')
    list_filter = ('status', 'ai_priority')
    search_fields = ('patient__username', 'assigned_doctor__username')
    readonly_fields = ('id', 'created_at', 'ai_processed_at')
    inlines = [CaseImageInline, VerdictInline]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('case', 'sender', 'created_at', 'read_at')
