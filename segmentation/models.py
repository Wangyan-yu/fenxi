from django.db import models
from django.utils import timezone


class CustomerData(models.Model):
    """客户数据模型"""
    customer_id = models.CharField(max_length=100, verbose_name="客户ID")
    age = models.IntegerField(null=True, blank=True, verbose_name="年龄")
    gender = models.CharField(max_length=10, null=True, blank=True, verbose_name="性别")
    annual_income = models.FloatField(null=True, blank=True, verbose_name="年收入(k$)")
    spending_score = models.IntegerField(null=True, blank=True, verbose_name="消费评分(1-100)")
    order_date = models.DateField(null=True, blank=True, verbose_name="订单日期")
    amount = models.FloatField(null=True, blank=True, verbose_name="订单金额")
    order_id = models.CharField(max_length=100, null=True, blank=True, verbose_name="订单ID")
    category = models.CharField(max_length=100, null=True, blank=True, verbose_name="商品类别")
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        verbose_name = "客户数据"
        verbose_name_plural = "客户数据"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.customer_id} - {self.age or 'N/A'}"


class AnalysisSession(models.Model):
    """分析会话模型"""
    session_name = models.CharField(max_length=200, verbose_name="会话名称")
    data_file = models.FileField(upload_to='uploads/', verbose_name="数据文件")
    analysis_type = models.CharField(max_length=50, choices=[
        ('clustering', '聚类分析'),
        ('rfm', 'RFM分析'),
        ('both', '两者都做'),
    ], default='both', verbose_name="分析类型")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "分析会话"
        verbose_name_plural = "分析会话"
        ordering = ['-created_at']

    def __str__(self):
        return self.session_name


class ClusteringResult(models.Model):
    """聚类结果模型"""
    session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='clustering_results', verbose_name="分析会话")
    customer_id = models.CharField(max_length=100, verbose_name="客户ID")
    cluster = models.IntegerField(verbose_name="聚类标签")
    r_score = models.IntegerField(null=True, blank=True, verbose_name="R评分")
    f_score = models.IntegerField(null=True, blank=True, verbose_name="F评分")
    m_score = models.IntegerField(null=True, blank=True, verbose_name="M评分")
    rfm_segment = models.CharField(max_length=100, null=True, blank=True, verbose_name="RFM细分")
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "聚类结果"
        verbose_name_plural = "聚类结果"
        ordering = ['cluster', 'customer_id']

    def __str__(self):
        return f"{self.customer_id} - 聚类 {self.cluster}"
