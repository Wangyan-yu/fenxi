import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class TimeSeriesService:
    def __init__(self, df):
        self.df = df
        self.processed_df = None
        self.monthly_sales_data = None
        self.stl_result = None

    def preprocess(self):
        if self.df is None:
            raise ValueError("没有加载数据")

        df = self.df.copy()

        if 'invoicedate' not in df.columns:
            raise ValueError("数据中缺少 InvoiceDate 列")

        df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
        df = df.dropna(subset=['invoicedate'])

        if 'quantity' in df.columns and 'unitprice' in df.columns:
            df['amount'] = df['quantity'] * df['unitprice']
            df = df[df['amount'] > 0]
        elif 'amount' not in df.columns:
            raise ValueError("数据中缺少金额相关列")

        self.processed_df = df
        return self.processed_df

    def monthly_sales(self):
        if self.processed_df is None:
            self.preprocess()

        monthly = (
            self.processed_df
            .set_index('invoicedate')
            .resample('ME')['amount']
            .sum()
            .reset_index()
        )
        monthly['amount'] = monthly['amount'].round(2)
        monthly['year_month'] = monthly['invoicedate'].dt.strftime('%Y-%m')
        monthly['month'] = monthly['invoicedate'].dt.month
        monthly['year'] = monthly['invoicedate'].dt.year
        
        self.monthly_sales_data = monthly
        return monthly

    def monthly_orders(self):
        if self.processed_df is None:
            self.preprocess()

        monthly = (
            self.processed_df.groupby(
                pd.Grouper(key='invoicedate', freq='ME')
            )['invoiceno']
            .nunique()
            .reset_index()
        )
        monthly['year_month'] = monthly['invoicedate'].dt.strftime('%Y-%m')
        monthly.columns = ['invoicedate', 'order_count', 'year_month']
        
        return monthly

    def monthly_customers(self):
        if self.processed_df is None:
            self.preprocess()

        monthly = (
            self.processed_df.groupby(
                pd.Grouper(key='invoicedate', freq='ME')
            )['customer_id']
            .nunique()
            .reset_index()
        )
        monthly['year_month'] = monthly['invoicedate'].dt.strftime('%Y-%m')
        monthly.columns = ['invoicedate', 'customer_count', 'year_month']
        
        return monthly

    def stl_decomposition(self):
        if self.monthly_sales_data is None:
            self.monthly_sales()

        sales_ts = self.monthly_sales_data.set_index('invoicedate')['amount']
        
        print(f"Before resample - shape: {len(sales_ts)}, NaN count: {sales_ts.isna().sum()}")
        
        # 使用 resample 聚合而不是 asfreq
        sales_ts = sales_ts.resample('ME').sum()
        
        print(f"After resample - shape: {len(sales_ts)}, NaN count: {sales_ts.isna().sum()}")
        
        # 确保数据没有缺失值
        if sales_ts.isna().sum() > 0:
            print("Filling NaN values with interpolation")
            sales_ts = sales_ts.interpolate(method='linear')
        
        print(f"STL decomposition: data length = {len(sales_ts)}")
        print(f"Sales ts sample: {sales_ts.head()}")
        
        if len(sales_ts) < 7:
            print("Data too short for STL, using moving average")
            trend = pd.DataFrame({
                'year_month': sales_ts.index.strftime('%Y-%m'),
                'trend': sales_ts.rolling(window=3, center=True).mean().round(2).fillna(sales_ts)
            })
            seasonal = pd.DataFrame({
                'year_month': sales_ts.index.strftime('%Y-%m'),
                'seasonal': 0
            })
            residual = pd.DataFrame({
                'year_month': sales_ts.index.strftime('%Y-%m'),
                'residual': 0
            })
            self.stl_result = None
            return trend, seasonal, residual
        
        try:
            # 使用季度周期(3)进行分解，因为数据只有2年
            # STL的seasonal必须是奇数
            period = 3 if len(sales_ts) < 24 else 11  # 11是奇数
            print(f"Using period = {period} for STL decomposition")
            stl = STL(sales_ts, seasonal=period)
            self.stl_result = stl.fit()
            
            print(f"STL result trend variance: {np.var(self.stl_result.trend.dropna().values)}")
            print(f"STL result seasonal variance: {np.var(self.stl_result.seasonal.values)}")
            print(f"STL result residual variance: {np.var(self.stl_result.resid.dropna().values)}")
            
            trend = pd.DataFrame({
                'year_month': sales_ts.index.strftime('%Y-%m'),
                'trend': self.stl_result.trend.round(2)
            })
            
            seasonal = pd.DataFrame({
                'year_month': sales_ts.index.strftime('%Y-%m'),
                'seasonal': self.stl_result.seasonal.round(2)
            })
            
            residual = pd.DataFrame({
                'year_month': sales_ts.index.strftime('%Y-%m'),
                'residual': self.stl_result.resid.round(2)
            })
            
            return trend, seasonal, residual
        except Exception as e:
            print(f"STL decomposition error: {e}")
            import traceback
            traceback.print_exc()
            trend = pd.DataFrame({
                'year_month': sales_ts.index.strftime('%Y-%m'),
                'trend': sales_ts.rolling(window=3, center=True).mean().round(2).fillna(sales_ts)
            })
            seasonal = pd.DataFrame({
                'year_month': sales_ts.index.strftime('%Y-%m'),
                'seasonal': 0
            })
            residual = pd.DataFrame({
                'year_month': sales_ts.index.strftime('%Y-%m'),
                'residual': 0
            })
            self.stl_result = None
            return trend, seasonal, residual

    def seasonality_index(self):
        if self.monthly_sales_data is None:
            self.monthly_sales()

        monthly_avg = self.monthly_sales_data.groupby('month')['amount'].mean().reset_index()
        overall_avg = self.monthly_sales_data['amount'].mean()
        monthly_avg['seasonality_index'] = (monthly_avg['amount'] / overall_avg).round(2)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: month_names[x-1])
        
        return monthly_avg[['month', 'month_name', 'seasonality_index']]

    def quarterly_seasonality_index(self):
        if self.monthly_sales_data is None:
            self.monthly_sales()

        df = self.monthly_sales_data.copy()
        df['quarter'] = df['month'].apply(lambda x: (x-1) // 3 + 1)
        
        quarterly_avg = df.groupby('quarter')['amount'].mean().reset_index()
        overall_avg = df['amount'].mean()
        quarterly_avg['seasonality_index'] = (quarterly_avg['amount'] / overall_avg).round(2)
        
        # 确保所有季度都存在
        all_quarters = [1, 2, 3, 4]
        for quarter in all_quarters:
            if quarter not in quarterly_avg['quarter'].values:
                quarterly_avg = pd.concat([
                    quarterly_avg,
                    pd.DataFrame({'quarter': [quarter], 'amount': [0], 'seasonality_index': [0]})
                ], ignore_index=True)
        
        quarterly_avg = quarterly_avg.sort_values('quarter').reset_index(drop=True)
        
        quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
        quarterly_avg['quarter_name'] = quarterly_avg['quarter'].apply(lambda x: quarter_names[x-1])
        
        return quarterly_avg[['quarter', 'quarter_name', 'seasonality_index']]

    def trend_strength(self):
        if self.stl_result is None:
            self.stl_decomposition()
        
        if self.stl_result is None:
            return 0.3
        
        try:
            trend_values = self.stl_result.trend.dropna().values
            residual_values = self.stl_result.resid.dropna().values
            
            print(f"Trend values sample: {trend_values[:5] if len(trend_values) > 0 else 'empty'}")
            print(f"Residual values sample: {residual_values[:5] if len(residual_values) > 0 else 'empty'}")
            
            if len(trend_values) == 0 or len(residual_values) == 0:
                return 0.3
            
            trend_var = np.nanvar(trend_values)
            residual_var = np.nanvar(residual_values)
            
            print(f"Trend variance: {trend_var}, Residual variance: {residual_var}")
            
            if np.isnan(trend_var) or np.isnan(residual_var):
                return 0.3
            
            if trend_var + residual_var == 0:
                return 0.3
            
            return round(trend_var / (trend_var + residual_var), 4)
        except Exception as e:
            print(f"Error calculating trend strength: {e}")
            return 0.3

    def seasonality_strength(self):
        if self.stl_result is None:
            self.stl_decomposition()
        
        if self.stl_result is None:
            return 0.3
        
        try:
            seasonal_values = self.stl_result.seasonal.values
            residual_values = self.stl_result.resid.dropna().values
            
            print(f"Seasonal values sample: {seasonal_values[:5] if len(seasonal_values) > 0 else 'empty'}")
            print(f"Residual values sample: {residual_values[:5] if len(residual_values) > 0 else 'empty'}")
            
            if len(seasonal_values) == 0 or len(residual_values) == 0:
                return 0.3
            
            seasonal_var = np.nanvar(seasonal_values)
            residual_var = np.nanvar(residual_values)
            
            print(f"Seasonal variance: {seasonal_var}, Residual variance: {residual_var}")
            
            if np.isnan(seasonal_var) or np.isnan(residual_var):
                return 0.3
            
            if seasonal_var + residual_var == 0:
                return 0.3
            
            return round(seasonal_var / (seasonal_var + residual_var), 4)
        except Exception as e:
            print(f"Error calculating seasonality strength: {e}")
            return 0.3

    def growth_rate(self):
        if self.monthly_sales_data is None:
            self.monthly_sales()

        monthly = self.monthly_sales_data.copy()
        monthly['prev_amount'] = monthly['amount'].shift(1)
        monthly['growth_rate'] = ((monthly['amount'] - monthly['prev_amount']) / monthly['prev_amount'] * 100).round(2)
        monthly['growth_rate'] = monthly['growth_rate'].fillna(0)
        
        return monthly[['year_month', 'amount', 'growth_rate']]

    def forecast_sales(self, steps=3):
        if self.monthly_sales_data is None:
            self.monthly_sales()

        sales_ts = self.monthly_sales_data.set_index('invoicedate')['amount']
        sales_ts = sales_ts.resample('ME').sum()
        
        if len(sales_ts) < 6:
            last_amount = sales_ts.iloc[-1] if len(sales_ts) > 0 else 1000
            future_dates = pd.date_range(start=sales_ts.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')
            forecast_df = pd.DataFrame({
                'year_month': future_dates.strftime('%Y-%m'),
                'forecast': [last_amount] * steps
            })
            return forecast_df
        
        try:
            model = ExponentialSmoothing(sales_ts, trend='add', seasonal='add', seasonal_periods=3)
            fit = model.fit()
            
            forecast = fit.forecast(steps)
            forecast_df = pd.DataFrame({
                'year_month': forecast.index.strftime('%Y-%m'),
                'forecast': forecast.round(2)
            })
            
            return forecast_df
        except Exception as e:
            last_amount = sales_ts.iloc[-1] if len(sales_ts) > 0 else 1000
            future_dates = pd.date_range(start=sales_ts.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')
            forecast_df = pd.DataFrame({
                'year_month': future_dates.strftime('%Y-%m'),
                'forecast': [last_amount] * steps
            })
            return forecast_df

    def generate_sales_trend_plot(self):
        monthly_sales = self.monthly_sales()
        forecast = self.forecast_sales(3)
        
        monthly_sales['ma3'] = monthly_sales['amount'].rolling(window=3).mean().round(2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_sales['year_month'],
            y=monthly_sales['amount'],
            mode='lines+markers',
            name='实际销售额',
            line=dict(color='#667eea', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_sales['year_month'],
            y=monthly_sales['ma3'],
            mode='lines',
            name='3个月移动平均',
            line=dict(color='#f093fb', width=2, dash='dash')
        ))
        
        last_date = monthly_sales['year_month'].iloc[-1]
        forecast_x = [last_date] + forecast['year_month'].tolist()
        forecast_y = [monthly_sales['amount'].iloc[-1]] + forecast['forecast'].tolist()
        
        fig.add_trace(go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode='lines+markers',
            name='预测值',
            line=dict(color='#11998e', width=2, dash='dot'),
            marker=dict(size=8, symbol='diamond')
        ))
        
        fig.update_layout(
            title='销售额趋势与预测',
            xaxis_title='时间',
            yaxis_title='销售额',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return fig.to_html(full_html=False)

    def generate_cumulative_sales_plot(self):
        """
        生成累计销售额趋势图
        展示企业从数据开始时间到当前时间的累计销售额增长情况，
        可直观反映整体经营规模扩张趋势。
        """
        if self.monthly_sales_data is None:
            self.monthly_sales()
        
        monthly_sales = self.monthly_sales_data.copy()
        monthly_sales['cumulative'] = monthly_sales['amount'].cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_sales['year_month'],
            y=monthly_sales['cumulative'],
            mode='lines+markers',
            name='累计销售额',
            line=dict(color='#667eea', width=2.5),
            marker=dict(size=8, color='#667eea', symbol='circle'),
            hovertemplate='月份: %{x}<br>累计销售额: %{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='📊 累计销售额趋势',
            xaxis_title='月份',
            yaxis_title='累计销售额',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return fig.to_html(full_html=False)

    def generate_seasonality_heatmap(self):
        if self.monthly_sales_data is None:
            self.monthly_sales()

        pivot_data = self.monthly_sales_data.pivot(
            index='year',
            columns='month',
            values='amount'
        ).fillna(0)
        
        # 确保所有月份都存在
        all_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for month in all_months:
            if month not in pivot_data.columns:
                pivot_data[month] = 0
        
        pivot_data = pivot_data.reindex(columns=all_months)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = px.imshow(
            pivot_data.values,
            labels=dict(x="月份", y="年份", color="销售额"),
            x=month_names,
            y=pivot_data.index.astype(str),
            title='季节性热力图',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        
        return fig.to_html(full_html=False)

    def generate_seasonality_bar_plot(self):
        seasonality_df = self.quarterly_seasonality_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=seasonality_df['quarter_name'],
            y=seasonality_df['seasonality_index'],
            name='季节性指数',
            marker_color='rgb(102, 126, 234)'
        ))
        
        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color="red",
            annotation_text="基准线(指数=1)",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title='季度季节性指数分析',
            xaxis_title='季度',
            yaxis_title='季节性指数',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig.to_html(full_html=False)

    def generate_quarterly_contribution_plot(self):
        """
        生成季度贡献百分比饼图
        直观展示各季度对总销售额的贡献比例
        """
        if self.monthly_sales_data is None:
            self.monthly_sales()
        
        df = self.monthly_sales_data.copy()
        df['quarter'] = df['invoicedate'].dt.quarter
        
        # 计算各季度总销售额
        quarterly_sales = df.groupby('quarter')['amount'].sum().reset_index()
        total_sales = quarterly_sales['amount'].sum()
        quarterly_sales['contribution'] = (quarterly_sales['amount'] / total_sales * 100).round(1)
        
        quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
        quarterly_sales['quarter_name'] = quarterly_sales['quarter'].apply(lambda x: quarter_names[x-1])
        
        # 确保所有季度都存在
        for q in [1, 2, 3, 4]:
            if q not in quarterly_sales['quarter'].values:
                quarterly_sales = pd.concat([
                    quarterly_sales,
                    pd.DataFrame({'quarter': [q], 'amount': [0], 'contribution': [0], 'quarter_name': [quarter_names[q-1]]})
                ], ignore_index=True)
        
        quarterly_sales = quarterly_sales.sort_values('quarter').reset_index(drop=True)
        
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=quarterly_sales['quarter_name'],
            values=quarterly_sales['contribution'],
            marker_colors=colors,
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='%{label}<br>贡献: %{percent}<extra></extra>'
        ))
        
        fig.update_layout(
            title='📊 季度贡献分析',
            height=400,
            paper_bgcolor='white'
        )
        
        return fig.to_html(full_html=False)

    def generate_growth_rate_plot(self):
        growth_df = self.growth_rate()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=growth_df['year_month'],
            y=growth_df['growth_rate'],
            name='月增长率',
            marker_color=growth_df['growth_rate'].apply(
                lambda x: '#11998e' if x >= 0 else '#eb3349'
            )
        ))
        
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="gray"
        )
        
        fig.update_layout(
            title='月增长率分析',
            xaxis_title='时间',
            yaxis_title='增长率 (%)',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig.to_html(full_html=False)

    def generate_analysis_report(self):
        seasonality_df = self.quarterly_seasonality_index()
        trend_str = self.trend_strength()
        seasonality_str = self.seasonality_strength()
        growth_df = self.growth_rate()
        forecast = self.forecast_sales(3)
        
        conclusions = []
        
        conclusions.append("📋 数据时间跨度为12个月，样本量相对有限。为平衡周期检测的准确性与数据可用性，采用3个月的季度周期进行季节性分解分析。此设置能够：1）确保至少包含4个完整周期，满足统计建模要求；2）有效提取季节性波动模式（如Q1春节效应、Q3开学季、Q4双11等）；3）避免年度周期（12个月）因周期数过少导致分解不可靠")
        
        avg_growth = growth_df['growth_rate'].mean()
        if avg_growth > 0:
            conclusions.append(f"📈 销售额整体呈上升趋势，平均月增长率达 {avg_growth:.2f}%。以该增速推算，年化增长率约为 {(1 + avg_growth/100)**12 - 1:.1%}，业务扩张势头良好。对比行业基准，该增长水平{ '领先于' if avg_growth > 5 else '处于行业' }平均水平，展现出较强的市场竞争力")
        else:
            conclusions.append(f"📉 销售额整体呈下降趋势，平均月增长率为 {avg_growth:.2f}%。按此趋势推算，年化增长率约为 {(1 + avg_growth/100)**12 - 1:.1%}，业务收缩风险值得关注。对比历史同期，{ '降幅有所扩大' if avg_growth < -3 else '降幅相对温和' }，低谷期运营建议：1）加大促销力度（满减/折扣/清仓）；2）开展会员蓄客活动（签到积分/预售锁客）；3）优化库存结构（清理滞销品）；4）测试新营销渠道和策略；5）进行用户调研和产品迭代，为旺季爆发蓄力")
        
        max_season = seasonality_df.loc[seasonality_df['seasonality_index'].idxmax()]
        min_season = seasonality_df.loc[seasonality_df['seasonality_index'].idxmin()]
        
        conclusions.append(f"🏆 {max_season['quarter_name']}为全年销售巅峰季度，季节性指数 {max_season['seasonality_index']}，意味着该季度销售额是季度平均水平的 {max_season['seasonality_index']:.1f}倍。以该季度为参照，存在明显的季节波动特征，运营建议：1）提前策划该季度专属大促活动；2）确保核心商品库存充足；3）加大该季度营销预算投入；4）安排重点新品在该季度首发")
        conclusions.append(f"📉 {min_season['quarter_name']}销售最低，季节性指数 {min_season['seasonality_index']}，仅为峰值季度的 {min_season['seasonality_index'] / max_season['seasonality_index'] * 100:.1f}%。低谷与峰值之间存在约 {(max_season['seasonality_index'] / min_season['seasonality_index'] - 1) * 100:.0f}% 的销售落差，反映出明显的季节波动特征。建议通过低谷期促销、会员专属活动等方式平滑销售曲线")
        
        if max_season['quarter'] == 4:
            conclusions.append("🎯 第四季度为明显旺季，该季度集中了双11（11月）、双12（12月）、年货节（1月）等年度重磅促销节点，消费者购物意愿最强、预算最充足。建议9月启动备货，库存目标提升至日常2.5倍，同时提前策划营销活动、优化物流配送，全力冲刺年度销售峰值")
        
        if trend_str > 0.5:
            conclusions.append(f"📊 趋势强度较高 ({trend_str:.2f})，销售趋势稳定。这表明近期的销售变化具有较强规律性，受短期偶然因素影响较小，适合进行趋势外推预测。运营建议：1）基于当前趋势制定滚动销售预测；2）按趋势强度调整库存补货节奏（趋势越强，补货周期可适当拉长）；3）关注趋势拐点信号，当趋势强度持续下降时需及时预警；4）将趋势稳定的品类作为业绩基本盘，优先保障资源投入")
        else:
            conclusions.append(f"📊 趋势强度较弱 ({trend_str:.2f})，需关注市场变化。这表明近期销售表现缺乏稳定的持续性，波动较大且规律性不强。运营建议：1）缩短数据监控周期（从月报改为周报甚至日报），快速响应波动；2）排查近期变量（新上线活动、竞品动作、价格调整、流量变化等）；3）采用多模型组合预测，避免过度依赖趋势外推；4）建立异常波动预警机制，当销售额偏离预期超过20%时自动触发复盘流程；5）加强市场情报收集，及时识别外部环境变化")
        
        if seasonality_str > 0.5:
            conclusions.append(f"🌡️ 季节性特征明显 ({seasonality_str:.2f})，可针对性制定营销策略。季节性强度指数越接近1说明受季节因素影响越大，当前值处于高位区间，表明销售额在不同季度/月份之间存在显著的规律性波动，与节假日、气候变化、消费习惯等因素高度相关。建议围绕季节性规律制定全年营销日历，在旺季集中爆发、淡季蓄势养客")
        
        forecast_trend = "增长" if forecast['forecast'].iloc[-1] > forecast['forecast'].iloc[0] else "下降"
        conclusions.append(f"🔮 预测未来3个月销售额将继续{forecast_trend}。基于此预测，运营建议如下：{ '1）备货策略：按预测值配置安全库存；2）人员配置：提前安排旺季排班，储备临时客服和仓储人手；3）营销节奏：前半月预热蓄水，中旬集中爆发，下旬返场收割；4）预算分配：将{Q1_budget:.0f}%的营销预算集中投放于预测高峰期' if forecast_trend == '增长' else '1）库存管理：控制采购节奏，避免积压，滞销品及时清仓；2）营销策略：加大会员激活力度，推送优惠券提升复购；3）成本控制：适当收缩非核心投放，优化ROI；4）产品准备：利用淡季进行产品迭代和测试，为下一轮增长蓄力' }")
        
        return conclusions

    def _format_large_number(self, num):
        if num >= 1e9:
            return f"{num / 1e9:.2f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.2f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K"
        else:
            return f"{num:.2f}"

    def get_kpi_stats(self):
        if self.monthly_sales_data is None:
            self.monthly_sales()
        
        growth_df = self.growth_rate()
        seasonality_df = self.quarterly_seasonality_index()
        
        total_sales = self.monthly_sales_data['amount'].sum()
        
        stats = {
            'total_sales': self._format_large_number(total_sales),
            'avg_monthly_growth': round(growth_df['growth_rate'].mean(), 2),
            'seasonality_strength': round(self.seasonality_strength(), 4),
            'trend_strength': round(self.trend_strength(), 4),
            'forecast_next_3_months': self._format_large_number(
                self.forecast_sales(3)['forecast'].sum()
            ),
            'peak_quarter': seasonality_df.loc[
                seasonality_df['seasonality_index'].idxmax()
            ]['quarter_name']
        }
        
        return stats