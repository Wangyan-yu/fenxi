from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_data, name='upload_data'),
    path('data-summary/', views.data_summary, name='data_summary'),
    path('clustering/', views.clustering_analysis, name='clustering_analysis'),
    path('clustering/perform/', views.perform_clustering, name='perform_clustering'),
    path('rfm-clustering/', views.rfm_clustering, name='rfm_clustering'),
    path('rfm-clustering/perform/', views.perform_rfm_clustering, name='perform_rfm_clustering'),
    path('time-series/', views.time_series_analysis, name='time_series_analysis'),
    path('download-sample/', views.download_sample_data, name='download_sample_data'),
]
