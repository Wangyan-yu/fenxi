from django import forms
from .models import AnalysisSession


class DataUploadForm(forms.Form):
    """数据上传表单"""
    data_file = forms.FileField(
        label="选择文件",
        help_text="支持 CSV, Excel 格式 (.csv, .xlsx, .xls)",
        widget=forms.FileInput(attrs={
            'accept': '.csv,.xlsx,.xls',
            'class': 'form-control form-control-lg',
            'id': 'data_file'
        })
    )
    
    def clean_data_file(self):
        file = self.cleaned_data['data_file']
        filename = file.name.lower()
        
        if not (filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls')):
            raise forms.ValidationError("只支持 CSV, Excel 格式的文件")
        
        if file.size > 10 * 1024 * 1024:  # 10MB
            raise forms.ValidationError("文件大小不能超过 10MB")
        
        return file


class ClusteringSetupForm(forms.Form):
    """聚类设置表单"""
    ALGORITHM_CHOICES = [
        ('kmeans', 'K-Means 聚类'),
        ('dbscan', 'DBSCAN 聚类'),
        ('hierarchical', '层次聚类'),
    ]
    
    SCALER_CHOICES = [
        ('standard', 'StandardScaler (标准化)'),
        ('minmax', 'MinMaxScaler (归一化)'),
    ]
    
    algorithm = forms.ChoiceField(
        choices=ALGORITHM_CHOICES,
        initial='kmeans',
        label="聚类算法",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    n_clusters = forms.IntegerField(
        initial=5,
        min_value=2,
        max_value=20,
        label="聚类数量 (K)",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    max_iter = forms.IntegerField(
        initial=300,
        min_value=100,
        max_value=1000,
        label="最大迭代次数",
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        required=False
    )
    
    random_state = forms.IntegerField(
        initial=42,
        label="随机种子",
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        required=False
    )
    
    use_r = forms.BooleanField(
        initial=True,
        label="使用 R (Recency) 特征",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        required=False
    )
    
    use_f = forms.BooleanField(
        initial=True,
        label="使用 F (Frequency) 特征",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        required=False
    )
    
    use_m = forms.BooleanField(
        initial=True,
        label="使用 M (Monetary) 特征",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        required=False
    )
    
    scaler = forms.ChoiceField(
        choices=SCALER_CHOICES,
        initial='standard',
        label="数据标准化方法",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
