# fenxi
# 🏬 商场顾客细分分析系统

基于 Django 的商场顾客细分分析系统，整合 **K-Means 聚类**与**时间序列季节性分析**，提供从数据上传、预处理到客户分群、销售预测的全链路解决方案。

---

## 🚀 功能特性

### 📤 顾客数据上传与管理
- **多格式支持**：支持 CSV、Excel、JSON 等常见数据格式
- **编码自动识别**：自动检测文件编码，避免乱码
- **多文件合并**：支持上传多个数据文件并自动合并
- **数据预览**：上传后即时预览数据表格与基础统计信息

### 🔍 K-Means 聚类分析
- **自动选 K**：利用肘部法则（Elbow Method）和轮廓系数自动推荐最优聚类数
- **智能预处理**：自动处理缺失值、异常值，并进行特征标准化
- **可视化展示**：提供聚类散点图、轮廓图、簇中心对比图
- **聚类描述**：自动生成每个簇的统计特征和业务解读

### 📈 时间序列季节性分析
- **趋势分解**：使用 STL 或经典分解法拆解趋势、季节和残差成分
- **销售预测**：基于 ARIMA 或指数平滑模型进行未来销售预测
- **季节性分析**：识别年度、季度、月度周期性规律
- **自动报告**：一键生成包含图表和统计指标的分析报告

### 📊 可视化展示与 RFM 分析
- **RFM 指标计算**：自动计算最近一次消费（Recency）、消费频率（Frequency）、消费金额（Monetary）
- **客户细分**：基于 RFM 得分进行客户分层（如重要价值客户、流失客户等）
- **可视化报表**：提供交互式图表（Plotly）和静态图表（Matplotlib）

### 📄 顾客细分报告生成
- 支持导出为 PDF 或 HTML 格式，包含聚类结果、RFM 分层、季节性分析结论与业务建议

---

## 🧰 技术栈

### 后端框架
- **Django 5.2.14**：高并发 Web 框架，提供 RESTful API 与后台管理

### 数据分析与机器学习
- **Pandas 2.2.2**：数据清洗、转换与聚合
- **NumPy 2.0.1**：高性能数值计算
- **scikit-learn 1.5.1**：K-Means 聚类、预处理与模型评估
- **statsmodels 0.14.x**：时间序列分解、ARIMA 等统计模型

### 数据可视化
- **Plotly**：交互式图表，用于 Web 端动态展示
- **Matplotlib** 与 **Seaborn**：静态图表生成，用于报告导出

### 前端
- **HTML5 / CSS3 / JavaScript**：构建用户界面
- **Plotly.js**：浏览器端交互图表渲染

---

## ⚙️ 快速开始

### 环境要求
- Python 3.10+
- pip 22+

### 安装与运行
```bash
# 克隆项目
git clone <your-repo-url>
cd 商场顾客细分分析系统

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 安装依赖
pip install -r requirements.txt

# 数据库迁移
python manage.py migrate

# 启动开发服务器
python manage.py runserver
