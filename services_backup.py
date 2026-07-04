import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
import io

class CustomerSegmentationService:
    def __init__(self):
        self.df = None
        self.cluster_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.optimal_k = 5
        self.x_column = None
        self.y_column = None

    def load_data(self, file_path):
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        
        for encoding in encodings:
            try:
                self.df = pd.read_csv(file_path, encoding=encoding)
                self._detect_columns()
                return self.df
            except (UnicodeDecodeError, ValueError):
                continue
        
        raise ValueError("无法识别文件编码，请确保文件为UTF-8或GBK编码")

    def _detect_columns(self):
        expected_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
        
        if all(col in self.df.columns for col in expected_cols):
            self.x_column = 'Annual Income (k$)'
            self.y_column = 'Spending Score (1-100)'
        else:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                if 'UnitPrice' in numeric_cols and 'Quantity' in numeric_cols:
                    self._aggregate_transaction_data()
                else:
                    self.x_column = numeric_cols[0]
                    self.y_column = numeric_cols[1]
            else:
                raise ValueError("数据集中没有足够的数值列进行聚类分析")

    def _aggregate_transaction_data(self):
        if 'CustomerID' not in self.df.columns:
            raise ValueError("交易数据需要包含 CustomerID 列")
        
        self.df['TotalAmount'] = self.df['UnitPrice'] * self.df['Quantity']
        
        customer_df = self.df.groupby('CustomerID').agg({
            'Quantity': 'sum',
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        
        customer_df.columns = ['CustomerID', 'TotalQuantity', 'TotalAmount', 'OrderCount']
        
        self.df = customer_df
        self.x_column = 'TotalAmount'
        self.y_column = 'OrderCount'

    def preprocess_data(self):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        df = self.df.copy()
        
        df[self.x_column] = df[self.x_column].fillna(df[self.x_column].median())
        df[self.y_column] = df[self.y_column].fillna(df[self.y_column].median())
        
        return df

    def find_optimal_k(self, max_k=10):
        df = self.preprocess_data()
        features = df[[self.x_column, self.y_column]].values
        features_scaled = self.scaler.fit_transform(features)
        
        inertia = []
        silhouette_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(features_scaled)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features_scaled, labels))
        
        self.optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        
        return {
            'inertia': inertia,
            'silhouette_scores': silhouette_scores,
            'optimal_k': self.optimal_k,
            'k_range': list(range(2, max_k + 1))
        }

    def perform_clustering(self, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.optimal_k
        
        df = self.preprocess_data()
        features = df[[self.x_column, self.y_column]].values
        features_scaled = self.scaler.fit_transform(features)
        
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        df['Cluster'] = self.cluster_model.fit_predict(features_scaled)
        
        pca_result = self.pca.fit_transform(features_scaled)
        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]
        
        self.df = df
        return df

    def get_cluster_statistics(self):
        if self.df is None or 'Cluster' not in self.df.columns:
            raise ValueError("No clustering results available. Call perform_clustering first.")
        
        agg_dict = {
            self.x_column: ['mean', 'min', 'max'],
            self.y_column: ['mean', 'min', 'max'],
            'CustomerID': 'count'
        }
        
        if 'Age' in self.df.columns:
            agg_dict['Age'] = ['mean', 'min', 'max']
        
        stats = self.df.groupby('Cluster').agg(agg_dict).round(2)
        
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        stats = stats.reset_index()
        
        return stats

    def generate_cluster_plot(self):
        if self.df is None or 'Cluster' not in self.df.columns:
            raise ValueError("No clustering results available.")
        
        hover_data = ['CustomerID']
        if 'Age' in self.df.columns:
            hover_data.append('Age')
        if 'Gender' in self.df.columns:
            hover_data.append('Gender')
        
        fig = px.scatter(
            self.df,
            x=self.x_column,
            y=self.y_column,
            color='Cluster',
            title=f'Customer Segmentation based on {self.x_column} and {self.y_column}',
            labels={
                self.x_column: self.x_column,
                self.y_column: self.y_column
            },
            color_discrete_sequence=px.colors.qualitative.Set1,
            hover_data=hover_data
        )
        
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig.to_html(full_html=False)

    def generate_pca_plot(self):
        if self.df is None or 'PCA1' not in self.df.columns:
            raise ValueError("No PCA results available.")
        
        hover_data = [self.x_column, self.y_column]
        if 'Age' in self.df.columns:
            hover_data.append('Age')
        
        fig = px.scatter(
            self.df,
            x='PCA1',
            y='PCA2',
            color='Cluster',
            title='PCA Visualization of Customer Segments',
            labels={
                'PCA1': 'Principal Component 1',
                'PCA2': 'Principal Component 2'
            },
            color_discrete_sequence=px.colors.qualitative.Set1,
            hover_data=hover_data
        )
        
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig.to_html(full_html=False)

    def generate_elbow_plot(self, elbow_data=None):
        if elbow_data is None:
            elbow_data = self.find_optimal_k()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=elbow_data['k_range'],
            y=elbow_data['inertia'],
            mode='lines+markers',
            name='Inertia'
        ))
        
        fig.update_layout(
            title='Elbow Method for Optimal K',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Inertia',
            width=800,
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig.to_html(full_html=False)

    def generate_silhouette_plot(self, elbow_data=None):
        if elbow_data is None:
            elbow_data = self.find_optimal_k()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=elbow_data['k_range'],
            y=elbow_data['silhouette_scores'],
            name='Silhouette Score'
        ))
        
        fig.update_layout(
            title='Silhouette Score for Different K Values',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Silhouette Score',
            width=800,
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig.to_html(full_html=False)

    def get_cluster_descriptions(self):
        if self.df is None or 'Cluster' not in self.df.columns:
            return {}
        
        descriptions = {}
        cluster_stats = self.get_cluster_statistics()
        
        x_mean_col = f'{self.x_column}_mean'
        y_mean_col = f'{self.y_column}_mean'
        
        for _, row in cluster_stats.iterrows():
            cluster_id = int(row['Cluster'])
            x_val = row[x_mean_col]
            y_val = row[y_mean_col]
            count = row['CustomerID_count']
            
            x_median = self.df[self.x_column].median()
            y_median = self.df[self.y_column].median()
            
            if x_val > x_median and y_val > y_median:
                description = f"High {self.x_column} - High {self.y_column} ({count} customers)"
            elif x_val > x_median and y_val < y_median:
                description = f"High {self.x_column} - Low {self.y_column} ({count} customers)"
            elif x_val < x_median and y_val > y_median:
                description = f"Low {self.x_column} - High {self.y_column} ({count} customers)"
            elif x_val < x_median and y_val < y_median:
                description = f"Low {self.x_column} - Low {self.y_column} ({count} customers)"
            else:
                description = f"Middle Group ({count} customers)"
            
            descriptions[cluster_id] = description
        
        return descriptions

def load_sample_data():
    sample_data = {
        'CustomerID': range(1, 201),
        'Annual Income (k$)': np.random.randint(15, 137, 200),
        'Spending Score (1-100)': np.random.randint(1, 100, 200),
        'Gender': np.random.choice(['Male', 'Female'], 200)
    }
    df = pd.DataFrame(sample_data)
    
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return buffer
