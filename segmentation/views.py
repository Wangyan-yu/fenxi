from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest
from django.core.files.storage import FileSystemStorage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import zipfile
import io
from segmentation.services import CustomerSegmentationService, load_sample_data
from .timeseries_service import TimeSeriesService

def index(request):
    return render(request, 'segmentation/index.html')

def upload_data(request):
    if request.method == 'POST' and request.FILES.getlist('data_files'):
        data_files = request.FILES.getlist('data_files')
        
        allowed_extensions = ['.csv', '.xlsx', '.xls']
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp1252']
        
        for data_file in data_files:
            if not any(data_file.name.lower().endswith(ext) for ext in allowed_extensions):
                return HttpResponseBadRequest(f"文件 {data_file.name} 格式不支持，仅支持 CSV 和 Excel 文件")
        
        fs = FileSystemStorage()
        
        # 删除旧文件
        if 'data_file' in request.session:
            old_file = request.session['data_file']
            if fs.exists(old_file):
                fs.delete(old_file)
        
        # 合并所有上传的文件
        dfs = []
        for data_file in data_files:
            filename = fs.save(data_file.name, data_file)
            file_path = fs.path(filename)
            
            try:
                if data_file.name.lower().endswith(('.xlsx', '.xls')):
                    xls = pd.ExcelFile(file_path)
                    sheets = xls.sheet_names
                    for sheet in sheets:
                        df = pd.read_excel(xls, sheet_name=sheet)
                        dfs.append(df)
                    xls.close()
                else:
                    df = None
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding)
                            break
                        except (UnicodeDecodeError, ValueError):
                            continue
                    
                    if df is None:
                        return HttpResponseBadRequest(f"无法识别文件 {data_file.name} 的编码，请确保文件为UTF-8或GBK编码")
                    dfs.append(df)
            except Exception as e:
                return HttpResponseBadRequest(f"读取文件 {data_file.name} 失败: {str(e)}")
        
        if not dfs:
            return HttpResponseBadRequest("没有成功读取任何文件")
        
        # 合并所有数据
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # 标准化列名
        merged_df.columns = [CustomerSegmentationService._standardize_col_name(None, col) for col in merged_df.columns]
        
        # 保存合并后的数据到 media 目录
        import uuid
        merged_filename = f"merged_data_{uuid.uuid4().hex[:8]}.csv"
        merged_path = fs.path(merged_filename)
        merged_df.to_csv(merged_path, index=False, encoding='utf-8')
        
        request.session.flush()
        request.session['data_file'] = merged_filename
        
        print(f"Uploaded {len(data_files)} files, merged shape: {merged_df.shape}")
        if 'invoicedate' in merged_df.columns:
            print(f"Date range: {merged_df['invoicedate'].min()} to {merged_df['invoicedate'].max()}")
        
        return render(request, 'segmentation/data_summary.html', {
            'columns': merged_df.columns.tolist(),
            'sample_data': merged_df.head(10).to_html(index=False),
            'row_count': len(merged_df),
            'column_count': len(merged_df.columns)
        })
    
    return render(request, 'segmentation/upload.html')

def clustering_analysis(request):
    if 'data_file' not in request.session:
        return render(request, 'segmentation/data_summary.html', {'error': 'Please upload data first'})
    
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
        return render(request, 'segmentation/data_summary.html', {'error': 'Please upload data first'})
    
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

def data_summary(request):
    if 'data_file' not in request.session:
        return render(request, 'segmentation/index.html', {'error': 'Please upload data first'})
    
    data_file = request.session['data_file']
    fs = FileSystemStorage()
    file_path = fs.path(data_file)
    
    try:
        service = CustomerSegmentationService()
        service.load_data(file_path)
        df = service.preprocess_data()
        
        return render(request, 'segmentation/data_summary.html', {
            'columns': df.columns.tolist(),
            'sample_data': df.head(10).to_html(index=False),
            'row_count': len(df),
            'column_count': len(df.columns)
        })
    except Exception as e:
        return HttpResponseBadRequest(f"Error loading data: {str(e)}")

def rfm_clustering(request):
    if 'data_file' not in request.session:
        return render(request, 'segmentation/data_summary.html', {'error': 'Please upload data first'})
    
    data_file = request.session['data_file']
    fs = FileSystemStorage()
    file_path = fs.path(data_file)
    
    try:
        service = CustomerSegmentationService()
        service.load_data(file_path)
        
        rfm_df = service.calculate_rfm()
        
        features = rfm_df[['recency', 'frequency', 'monetary']].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        inertia = []
        silhouette_scores = []
        for k in range(2, 9):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(features_scaled)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features_scaled, labels))
        
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        
        elbow_plot = go.Figure()
        elbow_plot.add_trace(go.Scatter(x=list(range(2, 9)), y=inertia, mode='lines+markers'))
        elbow_plot.update_layout(title='Elbow Method for RFM Data', xaxis_title='K', yaxis_title='Inertia', width=600, height=400)
        
        silhouette_plot = go.Figure()
        silhouette_plot.add_trace(go.Bar(x=list(range(2, 9)), y=silhouette_scores))
        silhouette_plot.update_layout(title='Silhouette Score for RFM Data', xaxis_title='K', yaxis_title='Score', width=600, height=400)
        
        return render(request, 'segmentation/rfm_clustering_setup.html', {
            'elbow_plot': elbow_plot.to_html(full_html=False),
            'silhouette_plot': silhouette_plot.to_html(full_html=False),
            'optimal_k': optimal_k
        })
    except Exception as e:
        return HttpResponseBadRequest(f"Error during RFM analysis: {str(e)}")

def perform_rfm_clustering(request):
    if 'data_file' not in request.session:
        return render(request, 'segmentation/data_summary.html', {'error': 'Please upload data first'})
    
    if request.method == 'POST':
        n_clusters = int(request.POST.get('n_clusters', 5))
        
        data_file = request.session['data_file']
        fs = FileSystemStorage()
        file_path = fs.path(data_file)
        
        try:
            service = CustomerSegmentationService()
            service.load_data(file_path)
            
            # 使用服务的优化聚类方法（包含异常值处理、log1p变换、RobustScaler）
            service.perform_clustering(n_clusters, use_rfm=True)
            
            # 使用PCA结果生成聚类图（因为聚类是在处理后的数据上进行的）
            cluster_plot = service.generate_cluster_plot()
            pca_plot = service.generate_pca_plot()
            stats = service.get_cluster_statistics()
            descriptions = service.get_cluster_descriptions()
            evaluation = service.evaluate_clustering()
            
            rfm_df = service.rfm_df
            
            return render(request, 'segmentation/rfm_clustering_results.html', {
                'cluster_plot': cluster_plot,
                'pca_plot': pca_plot,
                'stats_table': stats.to_html(index=False),
                'descriptions': descriptions,
                'n_clusters': n_clusters,
                'rfm_stats': {
                    'avg_recency': rfm_df['recency'].mean().round(2),
                    'avg_frequency': rfm_df['frequency'].mean().round(2),
                    'avg_monetary': rfm_df['monetary'].mean().round(2)
                },
                'evaluation': evaluation
            })
        except Exception as e:
            return HttpResponseBadRequest(f"Error performing RFM clustering: {str(e)}")
    
    return render(request, 'segmentation/rfm_clustering_setup.html')

def time_series_analysis(request):
    if 'data_file' not in request.session:
        return render(request, 'segmentation/data_summary.html', {'error': 'Please upload data first'})
    
    data_file = request.session['data_file']
    fs = FileSystemStorage()
    file_path = fs.path(data_file)
    
    try:
        service = CustomerSegmentationService()
        service.load_data(file_path)
        df = service.preprocess_data()
        
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        if 'invoicedate' in df.columns:
            print(f"Date range: {df['invoicedate'].min()} to {df['invoicedate'].max()}")
        
        ts_service = TimeSeriesService(df)
        
        kpi_stats = ts_service.get_kpi_stats()
        sales_trend_plot = ts_service.generate_sales_trend_plot()
        cumulative_sales_plot = ts_service.generate_cumulative_sales_plot()
        seasonality_heatmap = ts_service.generate_seasonality_heatmap()
        seasonality_bar_plot = ts_service.generate_quarterly_contribution_plot()
        growth_rate_plot = ts_service.generate_growth_rate_plot()
        conclusions = ts_service.generate_analysis_report()
        
        return render(request, 'segmentation/time_series.html', {
            'kpi_stats': kpi_stats,
            'sales_trend_plot': sales_trend_plot,
            'cumulative_sales_plot': cumulative_sales_plot,
            'seasonality_heatmap': seasonality_heatmap,
            'seasonality_bar_plot': seasonality_bar_plot,
            'growth_rate_plot': growth_rate_plot,
            'conclusions': conclusions
        })
    except Exception as e:
        return HttpResponseBadRequest(f"Error during time series analysis: {str(e)}")

def download_sample_data(request):
    """下载示例数据文件（打包成zip）"""
    # 获取项目根目录下的示例数据文件
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file1 = os.path.join(base_dir, 'online_retail_I(1).csv')
    file2 = os.path.join(base_dir, 'online_retail_II(1).csv')
    
    # 创建内存中的zip文件
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(file1):
            zf.write(file1, 'online_retail_I.csv')
        if os.path.exists(file2):
            zf.write(file2, 'online_retail_II.csv')
    
    buffer.seek(0)
    response = HttpResponse(buffer.getvalue(), content_type='application/zip')
    response['Content-Disposition'] = 'attachment; filename=sample_data.zip'
    return response
