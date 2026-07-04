import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.graph_objects as go
import plotly.express as px
import io
from datetime import datetime


class CustomerSegmentationService:
    def __init__(self):
        self.df = None
        self.rfm_df = None
        self.cluster_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.optimal_k = 5
        self.feature_columns = []

    def parse_uploaded_file(self, file):
        """解析上传的文件，支持多种编码"""
        filename = file.name.lower()
        
        if filename.endswith('.csv'):
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp1252']
            self.df = None
            
            for encoding in encodings:
                try:
                    file.seek(0)
                    self.df = pd.read_csv(file, encoding=encoding)
                    break
                except (UnicodeDecodeError, ValueError):
                    continue
            
            if self.df is None:
                raise ValueError("无法识别文件编码，请确保文件为UTF-8或GBK编码")
        elif filename.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(file)
        else:
            raise ValueError("不支持的文件格式")
        
        # 标准化列名
        self.df.columns = [self._standardize_col_name(col) for col in self.df.columns]
        
        return self.df

    def _standardize_col_name(self, col):
        """标准化列名"""
        col = col.strip().lower()
        col_mapping = {
            'customerid': 'customer_id',
            'customer id': 'customer_id',
            'annual income (k$)': 'annual_income',
            'annual income': 'annual_income',
            'spending score (1-100)': 'spending_score',
            'spending score': 'spending_score',
            'order_date': 'order_date',
            'order date': 'order_date',
            'amount': 'amount',
            'order_id': 'order_id',
            'order id': 'order_id',
            'category': 'category',
            # 交易数据列名
            'invoiceno': 'invoiceno',
            'invoice no': 'invoiceno',
            'invoice number': 'invoiceno',
            'invoice': 'invoiceno',
            'quantity': 'quantity',
            'unitprice': 'unitprice',
            'unit price': 'unitprice',
            'price': 'unitprice',
            'invoicedate': 'invoicedate',
            'invoice date': 'invoicedate',
            'invoice_date': 'invoicedate',
            'invoicedate': 'invoicedate',
            'stockcode': 'stockcode',
            'stock code': 'stockcode',
            'description': 'description',
            'country': 'country',
        }
        return col_mapping.get(col, col)

    def load_data(self, file_path):
        """加载数据，支持CSV和Excel文件（支持多sheet）"""
        self.df = None
        
        if file_path.lower().endswith(('.xlsx', '.xls')):
            try:
                xls = pd.ExcelFile(file_path)
                sheets = xls.sheet_names
                
                if len(sheets) == 1:
                    self.df = pd.read_excel(xls, sheet_name=sheets[0])
                else:
                    dfs = []
                    for sheet in sheets:
                        df = pd.read_excel(xls, sheet_name=sheet)
                        dfs.append(df)
                    self.df = pd.concat(dfs, ignore_index=True)
                
                xls.close()
            except Exception as e:
                raise ValueError(f"读取Excel文件失败: {str(e)}")
        else:
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(file_path, encoding=encoding)
                    break
                except (UnicodeDecodeError, ValueError):
                    continue
            
            if self.df is None:
                raise ValueError("无法识别文件编码，请确保文件为UTF-8或GBK编码")
        
        self.df.columns = [self._standardize_col_name(col) for col in self.df.columns]
        return self.df

    def preprocess_data(self):
        """数据预处理"""
        if self.df is None:
            raise ValueError("没有加载数据")
        
        df = self.df.copy()
        
        # 填充缺失值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        return df

    def get_data_summary(self):
        """获取数据汇总信息"""
        df = self.preprocess_data()
        
        summary = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': df.columns.tolist(),
            'sample_data': df.head(10).to_html(index=False, classes='table table-striped'),
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
        }
        
        # 计算统计指标
        if 'amount' in df.columns:
            summary['total_amount'] = df['amount'].sum()
            summary['avg_amount'] = df['amount'].mean()
            summary['max_amount'] = df['amount'].max()
            summary['min_amount'] = df['amount'].min()
        
        if 'customer_id' in df.columns:
            summary['customer_count'] = df['customer_id'].nunique()
        
        return summary

    def calculate_rfm(self):
        """计算 RFM 指标"""
        if self.df is None:
            raise ValueError("没有加载数据")
        
        df = self.df.copy()
        
        # 数据清洗 - 使用标准化后的列名 customer_id
        if 'customer_id' in df.columns:
            df = df.dropna(subset=['customer_id'])
            df = df[df['customer_id'] != 0]
        
        if 'quantity' in df.columns and 'unitprice' in df.columns:
            df = df[df['quantity'] > 0]
            df = df[df['unitprice'] > 0]
            df['amount'] = df['quantity'] * df['unitprice']
        
        # 转换时间
        if 'invoicedate' in df.columns:
            df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
            df = df.dropna(subset=['invoicedate'])
            
            snapshot_date = df['invoicedate'].max() + pd.Timedelta(days=1)
            
            self.rfm_df = df.groupby('customer_id').agg({
                'invoicedate': lambda x: (snapshot_date - x.max()).days,
                'invoiceno': 'nunique',
                'amount': 'sum'
            })
            self.rfm_df.columns = ['recency', 'frequency', 'monetary']
        elif 'customer_id' in df.columns:
            # 使用现有的 RFM 列或生成模拟数据
            if all(col in df.columns for col in ['recency', 'frequency', 'monetary']):
                self.rfm_df = df[['customer_id', 'recency', 'frequency', 'monetary']].set_index('customer_id')
            else:
                np.random.seed(42)
                self.rfm_df = pd.DataFrame({
                    'recency': np.random.randint(1, 365, len(df)),
                    'frequency': np.random.randint(1, 50, len(df)),
                    'monetary': np.random.uniform(100, 10000, len(df))
                }, index=df['customer_id'])
        else:
            raise ValueError("数据中缺少计算 RFM 所需的列")
        
        return self.rfm_df

    def score_rfm(self, rfm_df=None):
        """RFM 评分"""
        if rfm_df is None:
            rfm_df = self.calculate_rfm()
        
        df = rfm_df.copy()
        
        # 使用五分位数评分
        df['r_score'] = pd.qcut(df['recency'], q=5, labels=[5, 4, 3, 2, 1])
        df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        df['m_score'] = pd.qcut(df['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        
        # 转换为整数
        df['r_score'] = df['r_score'].astype(int)
        df['f_score'] = df['f_score'].astype(int)
        df['m_score'] = df['m_score'].astype(int)
        
        # 组合 RFM 评分
        df['rfm_score'] = df['r_score'].astype(str) + df['f_score'].astype(str) + df['m_score'].astype(str)
        
        self.rfm_df = df
        return df

    def segment_rfm(self, scored_df=None):
        """RFM 客户分类"""
        if scored_df is None:
            scored_df = self.score_rfm()
        
        df = scored_df.copy()
        
        def assign_segment(row):
            r, f, m = row['r_score'], row['f_score'], row['m_score']
            
            if r >= 4 and f >= 4 and m >= 4:
                return '重要价值客户'
            elif r >= 4 and f >= 4 and m < 4:
                return '重要保持客户'
            elif r >= 4 and f < 4 and m >= 4:
                return '重要发展客户'
            elif r < 4 and f >= 4 and m >= 4:
                return '重要挽留客户'
            elif r >= 4 and f >= 3 and m >= 3:
                return '一般价值客户'
            elif r >= 3 and f >= 3:
                return '一般保持客户'
            elif r >= 3:
                return '一般发展客户'
            else:
                return '需要关注客户'
        
        df['segment'] = df.apply(assign_segment, axis=1)
        self.rfm_df = df
        return df

    def generate_rfm_visualizations(self):
        """生成 RFM 可视化图表"""
        if self.rfm_df is None:
            self.segment_rfm()
        
        df = self.rfm_df
        
        visualizations = {}
        
        # 1. RFM 三维散点图
        fig_3d = px.scatter_3d(
            df, x='recency', y='frequency', z='monetary',
            color='segment',
            title='RFM 三维散点图',
            labels={
                'recency': 'Recency (天数)',
                'frequency': 'Frequency (次数)',
                'monetary': 'Monetary (金额)'
            },
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_3d.update_layout(height=600)
        visualizations['rfm_3d'] = fig_3d.to_html(full_html=False)
        
        # 2. 各维度分布直方图
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=df['recency'], name='Recency', opacity=0.7))
        fig_hist.add_trace(go.Histogram(x=df['frequency'], name='Frequency', opacity=0.7))
        fig_hist.add_trace(go.Histogram(x=df['monetary'], name='Monetary', opacity=0.7))
        fig_hist.update_layout(
            title='RFM 各维度分布',
            barmode='overlay',
            height=400
        )
        visualizations['rfm_hist'] = fig_hist.to_html(full_html=False)
        
        # 3. 客户类型饼图
        segment_counts = df['segment'].value_counts()
        fig_pie = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='RFM 客户细分分布',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(height=500)
        visualizations['segment_pie'] = fig_pie.to_html(full_html=False)
        
        # 4. RFM 热力图
        rfm_matrix = df.groupby(['r_score', 'f_score']).agg({
            'monetary': 'mean'
        }).reset_index()
        rfm_pivot = rfm_matrix.pivot(index='r_score', columns='f_score', values='monetary')
        
        fig_heatmap = px.imshow(
            rfm_pivot,
            labels=dict(x="F Score", y="R Score", color="Average Monetary"),
            x=rfm_pivot.columns,
            y=rfm_pivot.index,
            title='RFM 热力图 (平均消费金额)',
            color_continuous_scale='Viridis'
        )
        fig_heatmap.update_layout(height=500)
        visualizations['rfm_heatmap'] = fig_heatmap.to_html(full_html=False)
        
        return visualizations

    def get_rfm_statistics(self):
        """获取 RFM 统计信息"""
        if self.rfm_df is None:
            self.segment_rfm()
        
        df = self.rfm_df
        
        stats = {
            'avg_recency': round(df['recency'].mean(), 2),
            'avg_frequency': round(df['frequency'].mean(), 2),
            'avg_monetary': round(df['monetary'].mean(), 2),
            'segment_counts': df['segment'].value_counts().to_frame('count').to_html(classes='table table-striped'),
            'rfm_summary': df.groupby('segment').agg({
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': ['mean', 'count']
            }).round(2).to_html(classes='table table-striped')
        }
        
        return stats

    def _prepare_rfm_features(self, rfm_df):
        """
        准备 RFM 特征，使用稳健的数据预处理流程：
        1. 异常值裁剪（1% - 99%分位数）
        2. 偏态变换（log1p）
        3. 鲁棒缩放（RobustScaler）
        """
        rfm = rfm_df.copy()
        
        # 1. 异常值裁剪（使用1%-99%分位数）
        for col in ['recency', 'frequency', 'monetary']:
            lower = rfm[col].quantile(0.01)
            upper = rfm[col].quantile(0.99)
            rfm[col] = rfm[col].clip(lower, upper)
        
        # 2. 偏态变换（log1p 处理右偏分布）
        rfm['recency_log'] = np.log1p(rfm['recency'])
        rfm['frequency_log'] = np.log1p(rfm['frequency'])
        rfm['monetary_log'] = np.log1p(rfm['monetary'])
        
        # 3. 选择特征
        X = rfm[['recency_log', 'frequency_log', 'monetary_log']]
        
        # 4. 鲁棒缩放（对异常值更稳健）
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, scaler

    def find_optimal_k(self, max_k=8, features=None):
        """寻找最优 K 值"""
        df = self.preprocess_data()
        
        # 检测交易数据并计算 RFM
        has_transaction_data = all(col in df.columns for col in ['invoiceno', 'customer_id', 'quantity', 'unitprice', 'invoicedate'])
        has_customer_data = all(col in df.columns for col in ['annual_income', 'spending_score'])
        
        # 打印调试信息
        print(f"原始数据形状: {df.shape}")
        
        if has_transaction_data:
            print("检测到交易数据，计算 RFM...")
            if self.rfm_df is None:
                self.calculate_rfm()
            print(f"RFM 数据形状: {self.rfm_df.shape}")
            
            # 使用稳健的 RFM 预处理流程
            features_scaled, self.scaler = self._prepare_rfm_features(self.rfm_df)
            
            print("交易数据预处理：异常值裁剪 + log1p + RobustScaler")
            print("  * RFM 数据通常严重右偏")
            print("  * 异常值裁剪(1%-99%)减少极端值影响")
            print("  * log1p 压缩长尾，使分布更接近正态")
            print("  * RobustScaler 对异常值比 StandardScaler 更稳健")
            print("  * 这样能让 KMeans 的欧氏距离更合理")
            
            features = ['recency', 'frequency', 'monetary']
        elif has_customer_data:
            print("检测到顾客数据")
            features = ['annual_income', 'spending_score']
            feature_data = df[features].values
            
            # 商场顾客数据使用标准化
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(feature_data)
            
            print("顾客数据预处理：StandardScaler")
        else:
            print("未检测到标准数据格式，使用数值列")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if col not in ['customer_id']][:2]
            feature_data = df[features].values if features else df.iloc[:, :2].values
            
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(feature_data)
            
            print("通用数据预处理：StandardScaler")
        
        inertia = []
        silhouette_scores = []
        ch_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(features_scaled)
            inertia.append(kmeans.inertia_)
            
            # 计算轮廓系数（大数据集采样）
            if len(features_scaled) > 10000:
                sample_size = min(10000, len(features_scaled))
                indices = np.random.choice(len(features_scaled), sample_size, replace=False)
                silhouette_scores.append(silhouette_score(features_scaled[indices], labels[indices]))
            else:
                silhouette_scores.append(silhouette_score(features_scaled, labels))
            
            # 计算 Calinski-Harabasz 指数（数值越大聚类效果越好）
            ch_scores.append(calinski_harabasz_score(features_scaled, labels))
        
        # 综合考虑轮廓系数和 Calinski-Harabasz 指数选择最优 K
        # 轮廓系数越大越好，CH指数越大越好
        normalized_sil = (np.array(silhouette_scores) - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))
        normalized_ch = (np.array(ch_scores) - np.min(ch_scores)) / (np.max(ch_scores) - np.min(ch_scores))
        combined_score = 0.5 * normalized_sil + 0.5 * normalized_ch
        self.optimal_k = combined_score.argmax() + 2
        self.feature_columns = features
        
        print(f"最优 K 值选择 - 轮廓系数最大 K: {np.argmax(silhouette_scores) + 2}")
        print(f"最优 K 值选择 - CH 指数最大 K: {np.argmax(ch_scores) + 2}")
        print(f"最优 K 值选择 - 综合评分最大 K: {self.optimal_k}")
        
        return {
            'inertia': inertia,
            'silhouette_scores': silhouette_scores,
            'ch_scores': ch_scores,
            'optimal_k': self.optimal_k,
            'k_range': list(range(2, max_k + 1))
        }

    def perform_clustering(self, n_clusters=None, algorithm='kmeans', params=None, use_rfm=False):
        """执行聚类"""
        if n_clusters is None:
            n_clusters = self.optimal_k
        
        df = self.preprocess_data()
        
        # 检测交易数据并计算 RFM
        has_transaction_data = all(col in df.columns for col in ['invoiceno', 'customer_id', 'quantity', 'unitprice', 'invoicedate'])
        has_customer_data = all(col in df.columns for col in ['annual_income', 'spending_score'])
        
        # 确定使用的特征和预处理方式
        if use_rfm or has_transaction_data:
            if self.rfm_df is None:
                self.calculate_rfm()
            print(f"使用 RFM 数据进行聚类，数据形状: {self.rfm_df.shape}")
            
            # 打印 RFM 数据描述统计
            print("\n=== RFM 数据描述统计 ===")
            print(self.rfm_df.describe())
            
            # 使用稳健的 RFM 预处理流程
            features_scaled, self.scaler = self._prepare_rfm_features(self.rfm_df)
            
            print("\n交易数据预处理：异常值裁剪 + log1p + RobustScaler")
            
            features = ['recency', 'frequency', 'monetary']
        elif has_customer_data:
            print("使用顾客数据进行聚类")
            features = ['annual_income', 'spending_score']
            feature_data = df[features].values
            
            # 商场顾客数据使用标准化
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(feature_data)
            
            print("顾客数据预处理：StandardScaler")
        else:
            print("使用数值列进行聚类")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if col not in ['customer_id']][:2]
            feature_data = df[features].values if features else df.iloc[:, :2].values
            
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(feature_data)
            
            print("通用数据预处理：StandardScaler")
        
        self.feature_columns = features
        
        # 执行聚类
        if algorithm == 'kmeans':
            max_iter = params.get('max_iter', 300) if params else 300
            random_state = params.get('random_state', 42) if params else 42
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter, n_init='auto')
            labels = self.cluster_model.fit_predict(features_scaled)
        elif algorithm == 'dbscan':
            self.cluster_model = DBSCAN(eps=0.5, min_samples=5)
            labels = self.cluster_model.fit_predict(features_scaled)
        elif algorithm == 'hierarchical':
            self.cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = self.cluster_model.fit_predict(features_scaled)
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        # 添加聚类标签
        if self.rfm_df is not None:
            self.rfm_df['cluster'] = labels
            result_df = self.rfm_df
        else:
            df['cluster'] = labels
            result_df = df
        
        # PCA 降维（基于变换后的特征）
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)
        result_df['pca1'] = pca_result[:, 0]
        result_df['pca2'] = pca_result[:, 1]
        
        # 计算并打印轮廓系数
        silhouette_avg = silhouette_score(features_scaled, labels)
        
        # 打印三个关键指标
        print("\n=== 聚类评估关键指标 ===")
        print(f"Silhouette Score: {silhouette_avg}")
        print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        
        # 打印处理后的 RFM 数据描述统计
        if use_rfm or has_transaction_data:
            print("\n=== 处理后的 RFM 数据描述 ===")
            rfm_processed = pd.DataFrame(features_scaled, columns=['recency_scaled', 'frequency_scaled', 'monetary_scaled'])
            print(rfm_processed.describe())
        
        self.df = result_df
        return result_df

    def evaluate_clustering(self):
        """评估聚类效果 - 包含 Silhouette 和 Calinski-Harabasz 指数"""
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("没有聚类结果")
        
        # 获取特征数据
        if self.rfm_df is not None and 'recency' in self.rfm_df.columns:
            features_df = self.rfm_df[['recency', 'frequency', 'monetary']].copy()
            
            # RFM 数据需要先做对数变换（与 perform_clustering 保持一致）
            for col in features_df.columns:
                low = features_df[col].quantile(0.01)
                high = features_df[col].quantile(0.99)
                features_df[col] = features_df[col].clip(low, high)
            
            features = np.log1p(features_df).values
        else:
            df = self.preprocess_data()
            if self.feature_columns:
                features = df[self.feature_columns].values
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                features = df[numeric_cols[:2]].values
        
        features_scaled = self.scaler.transform(features)
        labels = self.df['cluster'].values
        
        # 计算评估指标
        try:
            silhouette_avg = silhouette_score(features_scaled, labels)
        except Exception as e:
            silhouette_avg = None
            print(f"计算轮廓系数失败: {e}")
        
        try:
            ch_score = calinski_harabasz_score(features_scaled, labels)
        except Exception as e:
            ch_score = None
            print(f"计算 Calinski-Harabasz 指数失败: {e}")
        
        # 计算每个聚类的统计信息
        cluster_stats = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            cluster_size = np.sum(mask)
            cluster_features = features_scaled[mask]
            if cluster_size > 1:
                cluster_inertia = np.sum((cluster_features - cluster_features.mean(axis=0)) ** 2)
            else:
                cluster_inertia = 0
            cluster_stats.append({
                'cluster': label,
                'size': cluster_size,
                'inertia': cluster_inertia
            })
        
        # 计算 Dunn 指数（内部指标，越小越好）
        try:
            dunn_score = self._calculate_dunn_index(features_scaled, labels)
        except Exception as e:
            dunn_score = None
            print(f"计算 Dunn 指数失败: {e}")
        
        # 生成评估报告
        report = {
            'silhouette_score': round(silhouette_avg, 4) if silhouette_avg is not None else None,
            'calinski_harabasz_score': round(ch_score, 2) if ch_score is not None else None,
            'dunn_index': round(dunn_score, 4) if dunn_score is not None else None,
            'n_clusters': len(unique_labels),
            'total_samples': len(labels),
            'cluster_stats': cluster_stats,
            'evaluation_summary': self._generate_evaluation_summary(silhouette_avg, ch_score)
        }
        
        return report
    
    def _calculate_dunn_index(self, features, labels):
        """
        计算 Dunn 指数（聚类有效性指标）
        Dunn 指数 = 最小簇间距离 / 最大簇内距离
        值越大表示聚类效果越好
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return None
        
        # 计算每个簇的中心点
        centers = []
        for label in unique_labels:
            mask = labels == label
            centers.append(features[mask].mean(axis=0))
        
        # 计算最小簇间距离
        min_inter_cluster_dist = float('inf')
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_inter_cluster_dist:
                    min_inter_cluster_dist = dist
        
        # 计算最大簇内距离
        max_intra_cluster_dist = 0
        for label in unique_labels:
            mask = labels == label
            cluster_points = features[mask]
            if len(cluster_points) > 1:
                # 计算簇内最大距离（直径）
                distances = np.sqrt(((cluster_points[:, np.newaxis] - cluster_points) ** 2).sum(axis=2))
                max_dist = distances.max()
                if max_dist > max_intra_cluster_dist:
                    max_intra_cluster_dist = max_dist
        
        if max_intra_cluster_dist == 0:
            return None
        
        return min_inter_cluster_dist / max_intra_cluster_dist
    
    def _generate_evaluation_summary(self, silhouette, ch_score):
        """生成聚类效果评估摘要"""
        summary = []
        
        # 轮廓系数评估（-1到1，越大越好）
        if silhouette is not None:
            if silhouette >= 0.7:
                summary.append(f"📊 轮廓系数: {round(silhouette, 4)} - 聚类效果优秀")
            elif silhouette >= 0.5:
                summary.append(f"📊 轮廓系数: {round(silhouette, 4)} - 聚类效果良好")
            elif silhouette >= 0.3:
                summary.append(f"📊 轮廓系数: {round(silhouette, 4)} - 聚类效果一般")
            else:
                summary.append(f"📊 轮廓系数: {round(silhouette, 4)} - 聚类效果较差")
        
        # Calinski-Harabasz 指数评估（越大越好）
        if ch_score is not None:
            if ch_score >= 1000:
                summary.append(f"📈 CH指数: {round(ch_score, 2)} - 聚类分离度很高")
            elif ch_score >= 500:
                summary.append(f"📈 CH指数: {round(ch_score, 2)} - 聚类分离度较高")
            elif ch_score >= 100:
                summary.append(f"📈 CH指数: {round(ch_score, 2)} - 聚类分离度一般")
            else:
                summary.append(f"📈 CH指数: {round(ch_score, 2)} - 聚类分离度较低")
        
        return ' | '.join(summary)

    def get_cluster_statistics(self):
        """获取聚类统计信息"""
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("没有聚类结果")
        
        df = self.df
        
        # 确定用于统计的列
        agg_dict = {}
        if self.rfm_df is not None:
            agg_dict = {
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': ['mean', 'count']
            }
            if 'segment' in df.columns:
                agg_dict['segment'] = lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['cluster', 'pca1', 'pca2', 'customer_id']:
                    agg_dict[col] = 'mean'
            agg_dict[df.columns[0]] = 'count'
        
        stats = df.groupby('cluster').agg(agg_dict).round(2)
        stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stats.columns]
        stats = stats.reset_index()
        
        return stats

    def generate_cluster_plot(self):
        """生成聚类散点图 - 使用原始特征（添加抖动效果）"""
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("没有聚类结果")
        
        df = self.df
        
        # 确定用于绘图的列
        x_col, y_col = None, None
        plot_df = None
        
        # 对于RFM数据，使用处理后的log特征进行绘图并添加抖动
        if self.rfm_df is not None:
            rfm_plot_df = df.copy()
            rfm_plot_df['frequency_log'] = np.log1p(df['frequency'])
            rfm_plot_df['monetary_log'] = np.log1p(df['monetary'])
            
            # 添加横向抖动，让重叠的点更容易区分
            np.random.seed(42)  # 设置随机种子保证可重复性
            rfm_plot_df['frequency_log_jitter'] = rfm_plot_df['frequency_log'] + np.random.normal(0, 0.15, len(rfm_plot_df))
            rfm_plot_df['monetary_log_jitter'] = rfm_plot_df['monetary_log'] + np.random.normal(0, 0.15, len(rfm_plot_df))
            
            x_col, y_col = 'frequency_log_jitter', 'monetary_log_jitter'
            title = 'RFM 聚类结果 (Frequency vs Monetary - log变换 + 抖动)'
            x_label = 'Frequency (log)'
            y_label = 'Monetary (log)'
            plot_df = rfm_plot_df
        elif 'annual_income' in df.columns and 'spending_score' in df.columns:
            # 添加抖动
            plot_df = df.copy()
            np.random.seed(42)
            plot_df['annual_income_jitter'] = plot_df['annual_income'] + np.random.normal(0, 0.8, len(plot_df))
            plot_df['spending_score_jitter'] = plot_df['spending_score'] + np.random.normal(0, 0.8, len(plot_df))
            
            x_col, y_col = 'annual_income_jitter', 'spending_score_jitter'
            title = '客户细分聚类结果'
            x_label = 'Annual Income (k$)'
            y_label = 'Spending Score (1-100)'
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            cols = [c for c in numeric_cols if c not in ['cluster', 'pca1', 'pca2']]
            if len(cols) >= 2:
                x_col, y_col = cols[0], cols[1]
                title = '客户细分聚类结果'
                x_label = x_col
                y_label = y_col
                plot_df = df
            else:
                x_col, y_col = 'pca1', 'pca2'
                title = '聚类结果 (PCA降维)'
                x_label = 'PCA Component 1'
                y_label = 'PCA Component 2'
                plot_df = df
        
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color='cluster',
            title=title,
            labels={
                x_col: x_label,
                y_col: y_label
            },
            color_discrete_sequence=px.colors.qualitative.Set1,
            hover_data=df.columns.tolist(),
            opacity=0.7,
            size_max=60
        )
        
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig.to_html(full_html=False)

    def generate_pca_plot(self):
        """生成 PCA 降维可视化"""
        if self.df is None or 'pca1' not in self.df.columns:
            raise ValueError("没有 PCA 结果")
        
        df = self.df
        
        fig = px.scatter(
            df,
            x='pca1',
            y='pca2',
            color='cluster',
            title='PCA 降维可视化',
            labels={
                'pca1': 'Principal Component 1',
                'pca2': 'Principal Component 2'
            },
            color_discrete_sequence=px.colors.qualitative.Set1,
            hover_data=df.columns.tolist()
        )
        
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig.to_html(full_html=False)

    def generate_elbow_plot(self, elbow_data=None):
        """生成肘部法则图"""
        if elbow_data is None:
            elbow_data = self.find_optimal_k()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=elbow_data['k_range'],
            y=elbow_data['inertia'],
            mode='lines+markers',
            name='Inertia',
            line=dict(width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title='肘部法则 - 确定最优聚类数量',
            xaxis_title='聚类数量 (K)',
            yaxis_title='Inertia (组内平方和)',
            width=800,
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig.to_html(full_html=False)

    def generate_silhouette_plot(self, elbow_data=None):
        """生成轮廓系数和 Calinski-Harabasz 指数图"""
        if elbow_data is None:
            elbow_data = self.find_optimal_k()
        
        fig = go.Figure()
        
        # 轮廓系数
        fig.add_trace(go.Bar(
            x=elbow_data['k_range'],
            y=elbow_data['silhouette_scores'],
            name='Silhouette Score',
            marker_color='rgb(55, 83, 109)',
            yaxis='y'
        ))
        
        # 如果有 CH 指数数据，添加到第二个 Y 轴
        if 'ch_scores' in elbow_data and elbow_data['ch_scores']:
            fig.add_trace(go.Scatter(
                x=elbow_data['k_range'],
                y=elbow_data['ch_scores'],
                name='Calinski-Harabasz',
                marker=dict(color='rgb(26, 118, 255)', size=10),
                line=dict(width=3),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title='聚类评估指标',
            xaxis_title='聚类数量 (K)',
            yaxis=dict(
                title=dict(
                    text='Silhouette Score',
                    font=dict(color='rgb(55, 83, 109)')
                ),
                tickfont=dict(color='rgb(55, 83, 109)')
            ),
            yaxis2=dict(
                title=dict(
                    text='Calinski-Harabasz',
                    font=dict(color='rgb(26, 118, 255)')
                ),
                tickfont=dict(color='rgb(26, 118, 255)'),
                overlaying='y',
                side='right'
            ),
            width=800,
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(x=0.8, y=1.0)
        )
        
        return fig.to_html(full_html=False)

    def generate_radar_plot(self):
        """生成雷达图展示各类特征"""
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("没有聚类结果")
        
        stats = self.get_cluster_statistics()
        
        # 确定雷达图的特征
        features = []
        for col in stats.columns:
            if col != 'cluster' and not col.endswith('_count'):
                features.append(col)
        
        if not features:
            return ""
        
        # 归一化数据用于雷达图
        stats_normalized = stats.copy()
        for feature in features:
            min_val = stats_normalized[feature].min()
            max_val = stats_normalized[feature].max()
            if max_val != min_val:
                stats_normalized[feature] = (stats_normalized[feature] - min_val) / (max_val - min_val)
        
        fig = go.Figure()
        
        for idx, row in stats_normalized.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[feature] for feature in features],
                theta=features,
                fill='toself',
                name=f'聚类 {int(row["cluster"])}'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title='各聚类特征雷达图',
            width=800,
            height=700
        )
        
        return fig.to_html(full_html=False)

    def get_cluster_descriptions(self):
        """获取聚类描述"""
        if self.df is None or 'cluster' not in self.df.columns:
            return {}

        stats = self.get_cluster_statistics()
        descriptions = {}

        for _, row in stats.iterrows():
            cluster_id = int(row['cluster'])
            customer_count = row.get(row.index[-1], 'N/A')

            # 生成详细描述
            desc_parts = []

            # 基础信息：客户数量
            desc_parts.append(f"📊 包含 {customer_count} 位客户")

            # 基于RFM特征生成详细描述
            if self.rfm_df is not None:
                r_mean = row.get('recency_mean', 0)
                f_mean = row.get('frequency_mean', 0)
                m_mean = row.get('monetary_mean', 0)

                # 添加具体的RFM数值描述
                desc_parts.append(f"📅 平均最近消费间隔: {r_mean:.1f}天")
                desc_parts.append(f"🛒 平均消费频率: {f_mean:.1f}次")
                desc_parts.append(f"💰 平均消费金额: ¥{m_mean:.2f}")

                # 客户类型详细描述
                if r_mean < 30 and f_mean > 10 and m_mean > 1000:
                    type_desc = "🏆 高价值活跃客户 - 消费频繁且金额高，近期有活跃行为，是品牌最优质的黄金客户群体，具有高忠诚度、强购买力、低流失风险三大特征"
                    desc_parts.append(type_desc)
                    desc_parts.append("💡 运营建议: 针对该群体投入高优先级运营资源 - VIP服务提升体验（专属客服/优先处理/定制推荐）；会员升级激励（根据消费阶梯设置升级礼包，提升arpu值）；推荐裂变机制（利用社交关系链实现低成本获客，推荐转化率预计提升30%+）")

                elif r_mean > 180:
                    type_desc = "😴 休眠客户 - 已超过6个月未产生消费行为，客户活跃度显著下降，存在较高的流失风险，需要及时通过营销手段进行唤醒和激活"
                    desc_parts.append(type_desc)
                    desc_parts.append("💡 运营建议: 发送唤醒专属优惠券（满减/折扣），推送回归礼包（积分补发+限时权益），开展限时折扣活动（7天倒计时促单），配合短信/邮件/Push多渠道触达，设置“老客回归”专属标签进行精准营销")

                elif m_mean > 2000:
                    type_desc = "💎 高消费客户 - 消费金额显著高于平均水平，具有较强的购买力和消费意愿，是品牌高端产品的主力消费群体，值得精细化运营维护"
                    desc_parts.append(type_desc)
                    desc_parts.append("💡 运营建议: 推送高端新品首发信息，提供大额订单分期免息服务，赠送消费积分加倍权益，邀请参与品牌高端品鉴会，设置消费达标送限量礼品机制")

                elif f_mean > 10:
                    type_desc = "🔄 高频消费客户 - 消费频次高，与品牌互动密切，具有较强的忠诚度和复购习惯，是品牌稳定的基础客户群体"
                    desc_parts.append(type_desc)
                    desc_parts.append("💡 运营建议: 提供会员等级保护机制（防降级），消费积分加速兑换，优先发货权益保障，设置签到打卡领积分活动，推送限时拼团优惠提升客单价")

                elif r_mean < 30:
                    type_desc = "🔥 活跃客户 - 近期有消费行为，处于活跃状态，是品牌最容易触达和转化的群体，适合进行新品推广和交叉销售"
                    desc_parts.append(type_desc)
                    desc_parts.append("💡 运营建议: 保持每日新品推送节奏，设置限时特价秒杀提醒，推出每日签到积分奖励，开展互动小游戏（抽奖/打卡）提升粘性，推送关联商品推荐")

                else:
                    type_desc = "👥 普通客户 - 消费行为和活跃度处于中等水平，具有较大的提升空间，适合通过精准营销逐步培养消费习惯"
                    desc_parts.append(type_desc)
                    desc_parts.append("💡 运营建议: 定期推送促销信息培养消费习惯，设置满额包邮门槛提升客单价，开展积分抵现活动促进复购，推送爆款商品+优惠券组合，逐步引导向高频/高消费转化")

            # 基于收入/消费特征生成详细描述（如果没有RFM数据）
            else:
                # 查找收入和消费相关字段
                income_col = [c for c in row.index if 'income' in c.lower()][0] if any(
                    'income' in c.lower() for c in row.index) else None
                spending_col = [c for c in row.index if 'spending' in c.lower()][0] if any(
                    'spending' in c.lower() for c in row.index) else None

                if income_col and spending_col:
                    income = row[income_col]
                    spending = row[spending_col]

                    desc_parts.append(f"💵 平均收入水平: {income:.1f}分位")
                    desc_parts.append(f"🛍️ 平均消费水平: {spending:.1f}分位")

                    if income > 70 and spending > 70:
                        type_desc = "👑 高收入高消费客户 - 收入高且消费意愿强，是核心高端客户群"
                        desc_parts.append(type_desc)
                        desc_parts.append("💡 建议: 推送高端定制产品、邀请参加线下品鉴会、提供私人管家服务")

                    elif income > 70 and spending < 40:
                        type_desc = "💰 高收入低消费客户 - 收入高但消费保守，有巨大消费潜力"
                        desc_parts.append(type_desc)
                        desc_parts.append("💡 建议: 分析消费痛点、推送高性价比产品、增加消费场景触点")

                    elif income < 40 and spending > 60:
                        type_desc = "🎯 低收入高消费客户 - 收入有限但消费意愿强烈，重视品质生活"
                        desc_parts.append(type_desc)
                        desc_parts.append("💡 建议: 提供分期付款、限时折扣优惠、积分抵现活动")

                    elif income < 40 and spending < 40:
                        type_desc = "📉 低收入低消费客户 - 收入和消费均处于较低水平，价格敏感度高"
                        desc_parts.append(type_desc)
                        desc_parts.append("💡 建议: 推送平价爆款商品、发放优惠券、团购拼单优惠")

                    else:
                        type_desc = "📈 中等水平客户 - 收入和消费处于中等水平，是主流消费群体"
                        desc_parts.append(type_desc)
                        desc_parts.append("💡 建议: 保持常规营销推送、提升服务质量、挖掘升级需求")

                else:
                    # 如果没有任何特征数据
                    desc_parts.append("📋 客户画像正在分析中...")

            descriptions[cluster_id] = ' | '.join(desc_parts)

        return descriptions

    def export_results(self, format='csv'):
        """导出结果"""
        if self.df is None:
            raise ValueError("没有数据可导出")
        
        buffer = io.BytesIO() if format != 'csv' else io.StringIO()
        
        if format == 'csv':
            self.df.to_csv(buffer, index=True, encoding='utf-8-sig')
        elif format == 'excel':
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                self.df.to_excel(writer, sheet_name='Results', index=True)
                if self.rfm_df is not None:
                    self.rfm_df.to_excel(writer, sheet_name='RFM_Analysis', index=True)
        
        buffer.seek(0)
        return buffer


def load_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    n_samples = 200
    
    sample_data = {
        'CustomerID': range(1, n_samples + 1),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 70, n_samples),
        'Annual Income (k$)': np.random.randint(15, 137, n_samples),
        'Spending Score (1-100)': np.random.randint(1, 100, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return buffer


def load_sample_rfm_data():
    """生成示例 RFM 交易数据"""
    np.random.seed(42)
    n_transactions = 1000
    customer_ids = np.random.randint(1, 201, n_transactions)
    
    # 生成日期范围
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start_date, end_date)
    
    sample_data = {
        'customer_id': customer_ids,
        'order_id': [f'ORD{i:06d}' for i in range(1, n_transactions + 1)],
        'order_date': np.random.choice(date_range, n_transactions),
        'amount': np.random.uniform(50, 5000, n_transactions).round(2),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Home'], n_transactions)
    }
    
    df = pd.DataFrame(sample_data)
    
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return buffer
