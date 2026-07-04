# Mall Customer Segmentation

基于Django的商场顾客细分分析系统，使用K-Means聚类和时间序列分析技术。

## 功能特性

- 顾客数据上传与管理：多格式支持、编码自动识别、多文件合并、数据预览
- K-Means 聚类分析：自动选 K、智能预处理、可视化展示、聚类描述
- 时间序列季节性分析：趋势分解、销售预测、季节性分析、自动报告
- 可视化展示分析结果，RFM 分析：RFM 指标计算、客户细分、可视化报表
- 顾客细分报告生成

## 技术栈

- **后端**: Django 5.2.14
- **数据分析**: Pandas 2.2.2
- **机器学习**: scikit-learn 1.5.1
- **数值计算**：NumPy 2.0.1
- **统计分析**：statsmodels 0.14.x
- **可视化/数据可视化**: Plotly, Matplotlib, Seaborn
- **前端**: HTML, CSS, JavaScript (Plotly.js)

## 项目结构
```bash
mall-seg/
├── MallSeg/                  # Django 项目配置
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py          # 项目配置文件
│   ├── urls.py              # 根 URL 配置
│   └── wsgi.py
├── segmentation/             # 核心应用模块
│   ├── templates/           # HTML 模板
│   │   └── segmentation/
│   │       ├── base.html           # 基础模板
│   │       ├── index.html          # 首页
│   │       ├── upload.html         # 数据上传页
│   │       ├── data_summary.html   # 数据摘要页
│   │       ├── clustering_setup.html      # 聚类设置页
│   │       ├── clustering_results.html    # 聚类结果页
│   │       ├── rfm_clustering_setup.html  # RFM 设置页
│   │       ├── rfm_clustering_results.html # RFM 结果页
│   │       └── time_series.html    # 时间序列分析页
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py             # 表单定义
│   ├── models.py           # 数据模型
│   ├── services.py         # 客户细分服务（聚类、RFM）
│   ├── timeseries_service.py # 时间序列分析服务
│   ├── urls.py             # 应用 URL 配置
│   └── views.py            # 视图控制器
├── notebooks/              # Jupyter 笔记本（探索性分析）
│   └── customer_segmentation_analysis.ipynb
├── media/                  # 上传文件存储目录
├── .gitignore
├── requirements.txt        # 依赖清单
├── manage.py              # Django 管理命令
└── README.md              # 项目说明文档
```
---

## 安装运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
python manage.py runserver
```

**访问应用**
打开浏览器访问 http://localhost:8000/

## 📖 使用指南

### 1. 数据上传

- 点击「数据上传」菜单
- 选择 CSV 或 Excel 文件（支持多文件同时上传）
- 系统自动识别文件编码并合并数据
- 查看数据摘要和样本数据

### 2. 客户聚类分析

- 点击「客户聚类」菜单
- 系统自动计算肘部法则和轮廓系数，推荐最优聚类数量
- 可手动调整聚类数量 K
- 查看聚类结果和客户群体描述

### 3. 时间序列分析

- 点击「时间序列分析」菜单
- 查看销售趋势、季节性特征和预测结果
- 获取自动生成的业务洞察和运营建议

---

## 🔧 核心模块说明

### CustomerSegmentationService (services.py)

**主要方法**:
- `load_data(file_path)` - 加载数据文件
- `preprocess_data()` - 数据预处理
- `calculate_rfm()` - 计算 RFM 指标
- `score_rfm()` - RFM 评分
- `segment_rfm()` - RFM 客户分类
- `find_optimal_k()` - 寻找最优聚类数量
- `perform_clustering()` - 执行聚类分析
- `generate_cluster_plot()` - 生成聚类可视化图表
- `get_cluster_descriptions()` - 获取聚类描述

### TimeSeriesService (timeseries_service.py)

**主要方法**:
- `preprocess()` - 时间序列数据预处理
- `monthly_sales()` - 月度销售额统计
- `stl_decomposition()` - STL 时间序列分解
- `seasonality_index()` - 季节性指数计算
- `trend_strength()` - 趋势强度计算
- `seasonality_strength()` - 季节性强度计算
- `forecast_sales()` - 销售预测
- `generate_analysis_report()` - 生成分析报告

---

## 📈 可视化图表

系统提供以下交互式可视化：

| 图表类型 | 说明 |
| :--- | :--- |
| 肘部法则图 | 确定最优聚类数量 |
| 轮廓系数图 | 评估聚类质量 |
| 聚类散点图 | 展示客户群体分布 |
| PCA 降维图 | 高维数据可视化 |
| RFM 三维散点图 | RFM 特征分布 |
| 客户类型饼图 | 客户细分比例 |
| RFM 热力图 | R-F 评分矩阵 |
| 销售趋势图 | 销售额变化趋势 |
| 季节性热力图 | 年月销售额矩阵 |
| 季度贡献图 | 各季度销售占比 |
| 增长率分析图 | 月度增长率变化 |

---

## 📝 API 接口

| 路径 | 方法 | 功能 |
| :--- | :--- | :--- |
| `/` | GET | 首页 |
| `/upload/` | GET/POST | 数据上传 |
| `/data-summary/` | GET | 数据摘要 |
| `/clustering/` | GET | 聚类设置 |
| `/clustering/perform/` | POST | 执行聚类 |
| `/rfm-clustering/` | GET | RFM 分析设置 |
| `/rfm-clustering/perform/` | POST | 执行 RFM 聚类 |
| `/time-series/` | GET | 时间序列分析 |
| `/download-sample/` | GET | 下载示例数据 |

---