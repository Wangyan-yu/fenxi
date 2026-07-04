from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_data, name='upload_data'),
    path('clustering/', views.clustering_analysis, name='clustering_analysis'),
    path('clustering/perform/', views.perform_clustering, name='perform_clustering'),
    path('download-sample/', views.download_sample_data, name='download_sample_data'),
]
