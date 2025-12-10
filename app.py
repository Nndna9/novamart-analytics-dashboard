import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Marketing Analytics Dashboard", layout="wide")

# Set the title of the dashboard
st.title("🎯 Marketing Analytics Dashboard - NovaMart")
st.markdown("*Masters of AI in Business Program | Data-Driven Marketing Intelligence*")

# Load data
@st.cache_data
def load_data():
    """Load all CSV files from the dataset folder"""
    try:
        campaign_data = pd.read_csv('campaign_performance.csv')
        customer_data = pd.read_csv('customer_data.csv')
        product_sales = pd.read_csv('product_sales.csv')
        lead_scoring_results = pd.read_csv('lead_scoring_results.csv')
        feature_importance = pd.read_csv('feature_importance.csv')
        learning_curve = pd.read_csv('learning_curve.csv')
        geographic_data = pd.read_csv('geographic_data.csv')
        channel_attribution = pd.read_csv('channel_attribution.csv')
        funnel_data = pd.read_csv('funnel_data.csv')
        customer_journey = pd.read_csv('customer_journey.csv')
        correlation_matrix = pd.read_csv('correlation_matrix.csv')
        
        return {
            "campaign_data": campaign_data,
            "customer_data": customer_data,
            "product_sales": product_sales,
            "lead_scoring_results": lead_scoring_results,
            "feature_importance": feature_importance,
            "learning_curve": learning_curve,
            "geographic_data": geographic_data,
            "channel_attribution": channel_attribution,
            "funnel_data": funnel_data,
            "customer_journey": customer_journey,
            "correlation_matrix": correlation_matrix
        }
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure all CSV files are in the same directory as app.py")
        return None

data = load_data()

if data is None:
    st.stop()

# ============================================================================
# SECTION 1: COMPARISON CHARTS
# ============================================================================

def chart_channel_performance(campaign_data):
    """1.1 Bar Chart - Channel Performance Comparison"""
    metric = st.selectbox("Select Metric", ["Revenue", "Conversions", "ROAS"], key="channel_metric")
    
    metric_col = metric.lower() if metric.lower() != "roas" else "roas"
    channel_performance = campaign_data.groupby('channel')[metric_col].sum().sort_values(ascending=True)
    
    fig = px.barh(channel_performance, 
                   title=f"Total {metric} by Marketing Channel",
                   labels={metric_col: metric, 'channel': 'Channel'},
                   color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def chart_regional_performance(campaign_data):
    """1.2 Grouped Bar Chart - Regional Performance by Quarter"""
    year = st.selectbox("Select Year", sorted(campaign_data['year'].unique()), key="region_year")
    
    filtered_data = campaign_data[campaign_data['year'] == year].groupby(['region', 'quarter'])['revenue'].sum().reset_index()
    
    fig = px.bar(filtered_data, x='region', y='revenue', color='quarter',
                 title=f"Revenue by Region and Quarter ({year})",
                 labels={'revenue': 'Revenue', 'region': 'Region', 'quarter': 'Quarter'},
                 barmode='group')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def chart_campaign_contribution(campaign_data):
    """1.3 Stacked Bar Chart - Campaign Type Contribution"""
    view_type = st.radio("View Type", ["Absolute Values", "100% Stacked"], key="campaign_view")
    
    monthly_data = campaign_data.groupby(['campaign_type', pd.to_datetime(campaign_data['date']).dt.to_period('M')])['spend'].sum().reset_index()
    monthly_data['date'] = monthly_data['date'].astype(str)
    
    if view_type == "100% Stacked":
        fig = px.bar(monthly_data, x='date', y='spend', color='campaign_type',
                     title="Campaign Type Contribution (100% Stacked)",
                     barnorm='percent')
    else:
        fig = px.bar(monthly_data, x='date', y='spend', color='campaign_type',
                     title="Campaign Type Contribution (Absolute Values)")
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SECTION 2: TEMPORAL CHARTS
# ============================================================================

def chart_revenue_trend(campaign_data):
    """2.1 Line Chart - Revenue Trend Over Time"""
    aggregation = st.selectbox("Aggregation Level", ["Daily", "Weekly", "Monthly"], key="trend_agg")
    
    campaign_data['date'] = pd.to_datetime(campaign_data['date'])
    
    if aggregation == "Daily":
        trend_data = campaign_data.groupby('date')['revenue'].sum().reset_index()
    elif aggregation == "Weekly":
        trend_data = campaign_data.set_index('date').resample('W')['revenue'].sum().reset_index()
    else:
        trend_data = campaign_data.set_index('date').resample('M')['revenue'].sum().reset_index()
    
    fig = px.line(trend_data, x='date', y='revenue',
                  title=f"Revenue Trend ({aggregation})",
                  markers=True)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def chart_cumulative_conversions(campaign_data):
    """2.2 Area Chart - Cumulative Conversions"""
    region_filter = st.multiselect("Filter by Region", campaign_data['region'].unique(), 
                                   default=campaign_data['region'].unique(), key="conv_region")
    
    filtered_data = campaign_data[campaign_data['region'].isin(region_filter)].sort_values('date')
    filtered_data['date'] = pd.to_datetime(filtered_data['date'])
    
    cumulative_data = filtered_data.groupby(['date', 'channel'])['conversions'].sum().reset_index()
    cumulative_data['cumulative_conversions'] = cumulative_data.groupby('channel')['conversions'].cumsum()
    
    fig = px.area(cumulative_data, x='date', y='cumulative_conversions', color='channel',
                  title="Cumulative Conversions by Channel")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SECTION 3: DISTRIBUTION CHARTS
# ============================================================================

def chart_age_distribution(customer_data):
    """3.1 Histogram - Customer Age Distribution"""
    bin_size = st.slider("Bin Size", 1, 10, 5, key="age_bins")
    
    fig = px.histogram(customer_data, x='age', nbins=bin_size,
                       title="Customer Age Distribution",
                       labels={'age': 'Age', 'count': 'Number of Customers'},
                       color_discrete_sequence=['#636EFA'])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def chart_ltv_by_segment(customer_data):
    """3.2 Box Plot - Lifetime Value by Customer Segment"""
    show_points = st.checkbox("Show Individual Points", key="ltv_points")
    
    fig = px.box(customer_data, x='segment', y='ltv',
                 title="Lifetime Value Distribution by Segment",
                 points="all" if show_points else None)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def chart_satisfaction_distribution(customer_data):
    """3.3 Violin Plot - Satisfaction Score Distribution"""
    fig = px.violin(customer_data, x='nps_category', y='satisfaction_score',
                    title="Satisfaction Score Distribution by NPS Category",
                    color='nps_category')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SECTION 4: RELATIONSHIP CHARTS
# ============================================================================

def chart_income_vs_ltv(customer_data):
    """4.1 Scatter Plot - Income vs. Lifetime Value"""
    fig = px.scatter(customer_data, x='income', y='ltv', color='segment',
                     title="Income vs. Lifetime Value by Segment",
                     hover_data=['customer_id', 'age'],
                     trendline="ols")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def chart_channel_performance_matrix(campaign_data):
    """4.2 Bubble Chart - Channel Performance Matrix"""
    channel_agg = campaign_data.groupby('channel').agg({
        'ctr': 'mean',
        'conversion_rate': 'mean',
        'spend': 'sum'
    }).reset_index()
    
    fig = px.scatter(channel_agg, x='ctr', y='conversion_rate', size='spend', color='channel',
                     title="Channel Performance Matrix (CTR vs. Conversion Rate)",
                     size_max=50,
                     hover_data={'ctr': ':.2%', 'conversion_rate': ':.2%', 'spend': ':$,.0f'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def chart_correlation_heatmap(correlation_matrix):
    """4.3 Heatmap - Correlation Matrix"""
    fig = px.imshow(correlation_matrix,
                    title="Correlation Matrix - Marketing Metrics",
                    color_continuous_scale='RdBu',
                    zmin=-1, zmax=1)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def chart_calendar_heatmap(campaign_data):
    """4.4 Calendar Heatmap - Daily Performance"""
    metric = st.selectbox("Select Metric", ["Revenue", "Conversions", "Clicks"], key="calendar_metric")
    
    campaign_data['date'] = pd.to_datetime(campaign_data['date'])
    campaign_data['day_of_week'] = campaign_data['date'].dt.day_name()
    campaign_data['week_num'] = campaign_data['date'].dt.isocalendar().week
    
    metric_col = metric.lower()
    calendar_data = campaign_data.groupby(['week_num', 'day_of_week'])[metric_col].sum().reset_index()
    
    fig = px.density_heatmap(campaign_data, x='date', y=metric_col,
                              title=f"Daily {metric} Heatmap",
                              nbinsx=30, nbinsy=1)
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SECTION 5: PART-TO-WHOLE CHARTS
# ============================================================================

def chart_attribution_donut(channel_attribution):
    """5.1 Donut Chart - Attribution Model Comparison"""
    model = st.selectbox("Select Attribution Model", 
                        ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based'],
                        key="attribution_model")
    
    attribution_data = channel_attribution.groupby('channel')[model].sum().reset_index()
    attribution_data = attribution_data.sort_values(model, ascending=False)
    
    fig = px.pie(attribution_data, values=model, names='channel',
                 title=f"{model.replace('_', ' ').title()} Attribution Model",
                 hole=0.3)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def chart_product_treemap(product_sales):
    """5.2 Treemap - Product Sales Hierarchy"""
    fig = px.treemap(product_sales, 
                     path=['category', 'subcategory', 'product'],
                     values='sales',
                     color='profit_margin',
                     color_continuous_scale='RdYlGn',
                     title="Product Sales Hierarchy (Size=Sales, Color=Profit Margin)")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def chart_customer_sunburst(customer_data):
    """5.3 Sunburst Chart - Customer Segmentation Breakdown"""
    sunburst_data = customer_data.groupby(['region', 'city_tier', 'segment']).size().reset_index(name='count')
    
    fig = px.sunburst(sunburst_data,
                      path=['region', 'city_tier', 'segment'],
                      values='count',
                      title="Customer Segmentation by Region, City Tier & Segment")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def chart_funnel(funnel_data):
    """5.4 Funnel Chart - Conversion Funnel"""
    fig = go.Figure(go.Funnel(
        y=funnel_data['stage'],
        x=funnel_data['visitors'],
        textposition="inside",
        textinfo="value+percent previous"
    ))
    fig.update_layout(title="Marketing Conversion Funnel", height=500)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SECTION 6: GEOGRAPHIC CHARTS
# ============================================================================

def chart_choropleth_map(geographic_data):
    """6.1 Choropleth Map - State-wise Revenue"""
    metric = st.selectbox("Select Metric", 
                         ['revenue', 'customers', 'market_penetration', 'yoy_growth'],
                         key="geo_metric")
    
    fig = px.choropleth(geographic_data,
                        locations="state",
                        z=metric,
                        hover_name="state",
                        color_continuous_scale="Viridis",
                        title=f"State-wise {metric.replace('_', ' ').title()}")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def chart_bubble_map(geographic_data):
    """6.2 Bubble Map - Store Performance"""
    fig = px.scatter_geo(geographic_data,
                         lat='latitude',
                         lon='longitude',
                         size='store_count',
                         color='satisfaction',
                         hover_name='state',
                         hover_data={'latitude': False, 'longitude': False,
                                    'store_count': True, 'satisfaction': ':.2f'},
                         title="Store Performance by Location",
                         size_max=50)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SECTION 7: ML MODEL EVALUATION CHARTS
# ============================================================================

def chart_confusion_matrix(lead_scoring_results):
    """7.1 Confusion Matrix - Lead Scoring Model"""
    threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, key="cm_threshold")
    
    predicted_class = (lead_scoring_results['predicted_probability'] >= threshold).astype(int)
    cm = confusion_matrix(lead_scoring_results['actual_converted'], predicted_class)
    
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Not Converted', 'Converted'],
                    y=['Not Converted', 'Converted'],
                    color_continuous_scale='Blues',
                    title=f"Confusion Matrix (Threshold: {threshold})")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def chart_roc_curve(lead_scoring_results):
    """7.2 ROC Curve - Model Performance"""
    fpr, tpr, thresholds = roc_curve(lead_scoring_results['actual_converted'], 
                                     lead_scoring_results['predicted_probability'])
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                            name=f'ROC Curve (AUC = {roc_auc:.3f})',
                            line=dict(color='#636EFA', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                            name='Random Classifier',
                            line=dict(color='gray', width=2, dash='dash')))
    fig.update_layout(title="ROC Curve - Lead Scoring Model",
                     xaxis_title="False Positive Rate",
                     yaxis_title="True Positive Rate",
                     height=400)
    st.plotly_chart(fig, use_container_width=True)


def chart_learning_curve(learning_curve_data):
    """7.3 Learning Curve - Model Diagnostics"""
    fig = px.line(learning_curve_data, x='training_set_size', 
                  y=['training_score', 'validation_score'],
                  title="Learning Curve - Training vs. Validation",
                  labels={'value': 'Score', 'training_set_size': 'Training Set Size',
                         'variable': 'Metric'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def chart_feature_importance(feature_importance):
    """7.4 Feature Importance - Model Interpretability"""
    feature_importance = feature_importance.sort_values('importance', ascending=True)
    
    fig = px.barh(feature_importance, x='importance', y='feature',
                  title="Feature Importance - Lead Scoring Model",
                  labels={'importance': 'Importance Score', 'feature': 'Feature'},
                  color='importance',
                  color_continuous_scale='Viridis')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("📊 Navigation")
pages = {
    "Executive Overview": "exec_overview",
    "Campaign Analytics": "campaign_analytics",
    "Customer Insights": "customer_insights",
    "Product Performance": "product_perf",
    "Geographic Analysis": "geographic",
    "Attribution & Funnel": "attribution",
    "ML Model Evaluation": "ml_eval"
}

selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# ============================================================================
# PAGE CONTENT
# ============================================================================

if selected_page == "Executive Overview":
    st.header("📈 Executive Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = data["campaign_data"]['revenue'].sum()
        st.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    
    with col2:
        total_conversions = data["campaign_data"]['conversions'].sum()
        st.metric("Total Conversions", f"{total_conversions:,.0f}")
    
    with col3:
        avg_roas = data["campaign_data"]['roas'].mean()
        st.metric("Average ROAS", f"{avg_roas:.2f}x")
    
    with col4:
        customer_count = len(data["customer_data"])
        st.metric("Total Customers", f"{customer_count:,.0f}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue Trend")
        chart_revenue_trend(data["campaign_data"])
    
    with col2:
        st.subheader("Channel Performance")
        chart_channel_performance(data["campaign_data"])


elif selected_page == "Campaign Analytics":
    st.header("📢 Campaign Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Channel Comparison", "Regional Performance", "Campaign Contribution", "Cumulative Conversions"])
    
    with tab1:
        st.subheader("Channel Performance Comparison")
        chart_channel_performance(data["campaign_data"])
    
    with tab2:
        st.subheader("Regional Performance by Quarter")
        chart_regional_performance(data["campaign_data"])
    
    with tab3:
        st.subheader("Campaign Type Contribution")
        chart_campaign_contribution(data["campaign_data"])
    
    with tab4:
        st.subheader("Cumulative Conversions")
        chart_cumulative_conversions(data["campaign_data"])
    
    st.divider()
    st.subheader("Daily Performance Heatmap")
    chart_calendar_heatmap(data["campaign_data"])


elif selected_page == "Customer Insights":
    st.header("👥 Customer Insights")
    
    tab1, tab2, tab3 = st.tabs(["Distribution", "Relationships", "Segmentation"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Age Distribution")
            chart_age_distribution(data["customer_data"])
        with col2:
            st.subheader("LTV by Segment")
            chart_ltv_by_segment(data["customer_data"])
        
        st.subheader("Satisfaction Distribution")
        chart_satisfaction_distribution(data["customer_data"])
    
    with tab2:
        st.subheader("Income vs. Lifetime Value")
        chart_income_vs_ltv(data["customer_data"])
        
        st.subheader("Channel Performance Matrix")
        chart_channel_performance_matrix(data["campaign_data"])
    
    with tab3:
        st.subheader("Customer Segmentation Breakdown")
        chart_customer_sunburst(data["customer_data"])


elif selected_page == "Product Performance":
    st.header("📦 Product Performance")
    
    st.subheader("Product Sales Hierarchy")
    chart_product_treemap(data["product_sales"])


elif selected_page == "Geographic Analysis":
    st.header("🗺️ Geographic Analysis")
    
    tab1, tab2 = st.tabs(["Choropleth Map", "Bubble Map"])
    
    with tab1:
        st.subheader("State-wise Revenue")
        chart_choropleth_map(data["geographic_data"])
    
    with tab2:
        st.subheader("Store Performance by Location")
        chart_bubble_map(data["geographic_data"])


elif selected_page == "Attribution & Funnel":
    st.header("🔗 Attribution & Funnel Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attribution Model Comparison")
        chart_attribution_donut(data["channel_attribution"])
    
    with col2:
        st.subheader("Conversion Funnel")
        chart_funnel(data["funnel_data"])
    
    st.divider()
    st.subheader("Correlation Matrix")
    chart_correlation_heatmap(data["correlation_matrix"])


elif selected_page == "ML Model Evaluation":
    st.header("🤖 ML Model Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        chart_confusion_matrix(data["lead_scoring_results"])
    
    with col2:
        st.subheader("ROC Curve")
        chart_roc_curve(data["lead_scoring_results"])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Learning Curve")
        chart_learning_curve(data["learning_curve"])
    
    with col2:
        st.subheader("Feature Importance")
        chart_feature_importance(data["feature_importance"])

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.divider()
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "This dashboard provides comprehensive insights into marketing performance, "
    "customer behavior, product sales, and ML model evaluation for NovaMart."
)
st.sidebar.markdown("---")
st.sidebar.markdown("*Masters of AI in Business Program*")