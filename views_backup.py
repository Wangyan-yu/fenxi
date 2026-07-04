from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest
from django.core.files.storage import FileSystemStorage
from segmentation.services import CustomerSegmentationService, load_sample_data

def index(request):
    return render(request, 'segmentation/index.html')

def upload_data(request):
    if request.method == 'POST' and request.FILES.get('data_file'):
        data_file = request.FILES['data_file']
        
        if not data_file.name.endswith('.csv'):
            return HttpResponseBadRequest("Only CSV files are allowed")
        
        fs = FileSystemStorage()
        filename = fs.save(data_file.name, data_file)
        file_path = fs.path(filename)
        
        try:
            service = CustomerSegmentationService()
            service.load_data(file_path)
            df = service.preprocess_data()
            
            request.session['data_file'] = filename
            
            return render(request, 'segmentation/data_summary.html', {
                'columns': df.columns.tolist(),
                'sample_data': df.head(10).to_html(index=False),
                'row_count': len(df),
                'column_count': len(df.columns)
            })
        except Exception as e:
            return HttpResponseBadRequest(f"Error processing file: {str(e)}")
    
    return render(request, 'segmentation/upload.html')

def clustering_analysis(request):
    if 'data_file' not in request.session:
        return render(request, 'segmentation/index.html', {'error': 'Please upload data first'})
    
    data_file = request.session['data_file']
    fs = FileSystemStorage()
    file_path = fs.path(data_file)
    
    try:
        service = CustomerSegmentationService()
        service.load_data(file_path)
        
        elbow_data = service.find_optimal_k()
        elbow_plot = service.generate_elbow_plot(elbow_data)
        silhouette_plot = service.generate_silhouette_plot(elbow_data)
        optimal_k = elbow_data['optimal_k']
        
        return render(request, 'segmentation/clustering_setup.html', {
            'elbow_plot': elbow_plot,
            'silhouette_plot': silhouette_plot,
            'optimal_k': optimal_k
        })
    except Exception as e:
        return HttpResponseBadRequest(f"Error during analysis: {str(e)}")

def perform_clustering(request):
    if 'data_file' not in request.session:
        return render(request, 'segmentation/index.html', {'error': 'Please upload data first'})
    
    if request.method == 'POST':
        n_clusters = int(request.POST.get('n_clusters', 5))
        
        data_file = request.session['data_file']
        fs = FileSystemStorage()
        file_path = fs.path(data_file)
        
        try:
            service = CustomerSegmentationService()
            service.load_data(file_path)
            service.perform_clustering(n_clusters)
            
            cluster_plot = service.generate_cluster_plot()
            pca_plot = service.generate_pca_plot()
            cluster_stats = service.get_cluster_statistics().to_html(index=False)
            cluster_descriptions = service.get_cluster_descriptions()
            
            return render(request, 'segmentation/clustering_results.html', {
                'cluster_plot': cluster_plot,
                'pca_plot': pca_plot,
                'stats_table': cluster_stats,
                'descriptions': cluster_descriptions,
                'optimal_k': n_clusters
            })
        except Exception as e:
            return HttpResponseBadRequest(f"Error performing clustering: {str(e)}")
    
    return render(request, 'segmentation/clustering_setup.html')

def download_sample_data(request):
    buffer = load_sample_data()
    response = HttpResponse(buffer.getvalue(), content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=sample_customer_data.csv'
    return response
