from django.contrib import admin
from .models import CustomerData, AnalysisSession, ClusteringResult


@admin.register(CustomerData)
class CustomerDataAdmin(admin.ModelAdmin):
    list_display = ('customer_id', 'age', 'gender', 'annual_income', 'spending_score', 'created_at')
    list_filter = ('gender', 'created_at')
    search_fields = ('customer_id',)
    readonly_fields = ('created_at', 'updated_at')


@admin.register(AnalysisSession)
class AnalysisSessionAdmin(admin.ModelAdmin):
    list_display = ('session_name', 'analysis_type', 'created_at')
    list_filter = ('analysis_type', 'created_at')
    search_fields = ('session_name',)
    readonly_fields = ('created_at',)


@admin.register(ClusteringResult)
class ClusteringResultAdmin(admin.ModelAdmin):
    list_display = ('session', 'customer_id', 'cluster', 'r_score', 'f_score', 'm_score', 'rfm_segment', 'created_at')
    list_filter = ('cluster', 'r_score', 'f_score', 'm_score', 'rfm_segment', 'created_at')
    search_fields = ('customer_id',)
    readonly_fields = ('created_at',)
