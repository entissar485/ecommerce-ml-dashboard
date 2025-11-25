import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import joblib
import numpy as np

st.set_page_config(
    page_title="Análisis de Ventas E-Commerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main, .stApp, [data-testid="stSidebar"], section, div {
        background-color: #ffffff !important;
    }
    
    .stSelectbox > div > div,
    .stSelectbox [data-baseweb="select"],
    .stSelectbox [data-baseweb="popover"],
    .stSelectbox ul,
    .stSelectbox li,
    .stMultiSelect > div > div,
    .stMultiSelect [data-baseweb="select"],
    .stMultiSelect [data-baseweb="popover"],
    .stMultiSelect ul,
    .stMultiSelect li,
    [role="listbox"],
    [role="option"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #f4a5c4 !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    .stNumberInput input {
        background-color: #ffffff !important;
        border: 2px solid #f4a5c4 !important;
        color: #000000 !important;
    }
    
    body, p, span, div, label, h1, h2, h3, h4, h5, h6,
    .stMarkdown, [data-testid="stSidebar"] *,
    .stSelectbox *, .stMultiSelect *, .stNumberInput *,
    [role="option"], [role="listbox"] * {
        color: #000000 !important;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #c44e81 !important;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #000000 !important;
        margin-bottom: 3rem;
        text-align: center;
        font-weight: 600 !important;
    }
    
    .section-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #c44e81 !important;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        border-bottom: 4px solid #f4a5c4;
        padding-bottom: 0.8rem;
    }
    
    .sidebar-section {
        background-color: #fff5f9 !important;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        border: 2px solid #f4a5c4;
    }
    
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #c44e81 !important;
        margin-bottom: 1rem;
    }
    
    [data-testid="stSidebar"] {
        border-right: 3px solid #f4a5c4!important;
    }
    
    .stButton>button {
        background-color: #d4568c !important;
        color: none !important;
        border: none !important;
        border-radius: 0.8rem;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .stButton>button:hover {
        background-color: #b44676 !important;
        color: #000000 !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #c44e81 !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    .info-box, .metric-card {
        background-color: #fff5f9 !important;
        padding: 2rem;
        border-radius: 1.2rem;
        border: 2px solid #f4a5c4;
    }
    
    .info-box *, .metric-card * {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #d4568c !important;
    }
    
    .stSlider [data-testid="stTickBar"] {
        background: linear-gradient(90deg, #f4a5c4 0%, #d4568c 100%) !important;
    }
    
    .stCheckbox label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    .stSuccess, .stInfo {
        background-color: #fff5f9 !important;
        border: 2px solid #f4a5c4 !important;
    }
    
    .stSuccess *, .stInfo * {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

load_dotenv()

@st.cache_resource
def get_mongo_connection():
    # Intentar cargar desde Streamlit secrets primero (para deployment)
    try:
        MONGO_URI = st.secrets["MONGO_URI"]
        DB_NAME = st.secrets["DB_NAME"]
    except:
        # Si falla, usar variables de entorno locales
        MONGO_URI = os.getenv('MONGO_URI')
        DB_NAME = os.getenv('DB_NAME')
    
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db

@st.cache_data
def load_data():
    db = get_mongo_connection()
    collection = db['monthly_sales']
    data = list(collection.find())
    df = pd.DataFrame(data)
    df = df.drop('_id', axis=1)
    return df

@st.cache_resource
def load_models():
    gb_model = joblib.load('models/gradient_boosting_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le = joblib.load('models/label_encoder.pkl')
    return gb_model, scaler, le

try:
    df = load_data()
    gb_model, scaler, le = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error al cargar datos o modelos: {str(e)}")
    models_loaded = False

if models_loaded:
    st.markdown('<p class="main-header">Panel de Análisis de Ventas E-Commerce</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análisis predictivo avanzado con inteligencia artificial y datos en tiempo real</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-title">Período de Tiempo</p>', unsafe_allow_html=True)
        year_filter = st.multiselect(
            "Seleccione los años:",
            options=sorted(df['year'].unique()),
            default=sorted(df['year'].unique())
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-title">Categorías de Productos</p>', unsafe_allow_html=True)
        all_categories = sorted(df['category'].dropna().unique())
        
        select_all = st.checkbox("Seleccionar todas las categorías", value=False)
        
        if select_all:
            category_filter = all_categories
        else:
            category_filter = st.multiselect(
                "Seleccione las categorías:",
                options=all_categories,
                default=all_categories[:5] if len(all_categories) > 5 else all_categories
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-title">Información del Sistema</p>', unsafe_allow_html=True)
        st.markdown(f"""
        **Registros Totales:** {len(df):,}
        
        **Modelo:** Gradient Boosting
        
        **Precisión:** R² = 0.803
        
        **Base de Datos:** MongoDB Atlas (Azure)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    df_filtered = df[
        (df['year'].isin(year_filter)) & 
        (df['category'].isin(category_filter))
    ]
    
    st.markdown('<p class="section-header">Resumen Ejecutivo</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = df_filtered['total_sales'].sum()
        st.metric("Ingresos Totales", f"${total_sales:,.2f}")
        
    with col2:
        total_orders = df_filtered['orders_count'].sum()
        st.metric("Total de Órdenes", f"{total_orders:,}")
    
    with col3:
        avg_ticket = total_sales / total_orders if total_orders > 0 else 0
        st.metric("Ticket Promedio", f"${avg_ticket:.2f}")
    
    with col4:
        total_products = df_filtered['product_id'].nunique()
        st.metric("Productos Únicos", f"{total_products:,}")
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_freight = df_filtered['avg_freight'].mean()
        st.markdown(f"**Costo Promedio de Envío:** ${avg_freight:.2f}")
    with col2:
        avg_weight = df_filtered['product_weight'].mean()
        st.markdown(f"**Peso Promedio de Producto:** {avg_weight:.0f}g")
    with col3:
        total_categories = df_filtered['category'].nunique()
        st.markdown(f"**Categorías Activas:** {total_categories}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">Análisis de Desempeño de Ventas</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top 10 Categorías por Ingresos")
        
        sales_by_category = df_filtered.groupby('category')['total_sales'].sum().sort_values(ascending=False).head(10)
        
        fig1 = go.Figure(go.Bar(
            x=sales_by_category.values,
            y=sales_by_category.index,
            orientation='h',
            marker=dict(
                color=sales_by_category.values,
                colorscale=['#ffd4e5', '#ffb3d9', '#ff8fc7', '#f06aa7', '#d4568c'],
                line=dict(color='#c44e81', width=2)
            ),
            text=[f'${val:,.0f}' for val in sales_by_category.values],
            textposition='outside',
            textfont=dict(size=16, color='#000000', family='Arial Black')
        ))
        
        fig1.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=15, color='#000000', family='Arial'),
            xaxis=dict(
                title=dict(text="Ventas Totales (USD)", font=dict(size=16, color='#000000', family='Arial Black')),
                showgrid=True, 
                gridcolor='#e0e0e0',
                color='#000000',
                tickfont=dict(color='#000000', size=14, family='Arial')
            ),
            yaxis=dict(
                title=dict(text="Categoría", font=dict(size=16, color='#000000', family='Arial Black')),
                showgrid=False, 
                color='#000000',
                tickfont=dict(color='#000000', size=14, family='Arial')
            )
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### Tendencia de Ventas Mensuales")
        
        df_temp = df_filtered.copy()
        df_temp['date'] = pd.to_datetime(df_temp['year'].astype(str) + '-' + df_temp['month'].astype(str) + '-01')
        sales_trend = df_temp.groupby('date')['total_sales'].sum().reset_index()
        
        fig2 = go.Figure(go.Scatter(
            x=sales_trend['date'],
            y=sales_trend['total_sales'],
            mode='lines+markers',
            line=dict(color='#d4568c', width=4),
            marker=dict(size=10, color='#f06aa7', line=dict(color='#c44e81', width=2)),
            fill='tozeroy',
            fillcolor='rgba(255, 179, 217, 0.2)',
            text=[f'${val:,.0f}' for val in sales_trend['total_sales']],
            hovertemplate='<b>Fecha:</b> %{x}<br><b>Ventas:</b> %{text}<extra></extra>'
        ))
        
        fig2.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=15, color='#000000', family='Arial'),
            xaxis=dict(
                title=dict(text="Fecha", font=dict(size=16, color='#000000', family='Arial Black')),
                showgrid=True, 
                gridcolor='#e0e0e0',
                color='#000000',
                tickfont=dict(color='#000000', size=14, family='Arial')
            ),
            yaxis=dict(
                title=dict(text="Ventas (USD)", font=dict(size=16, color='#000000', family='Arial Black')),
                showgrid=True, 
                gridcolor='#e0e0e0',
                color='#000000',
                tickfont=dict(color='#000000', size=14, family='Arial')
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Distribución de Ventas")
        
        fig3 = go.Figure(go.Histogram(
            x=df_filtered['total_sales'],
            nbinsx=50,
            marker=dict(
                color='#d4568c',
                line=dict(color='#c44e81', width=1.5)
            ),
            opacity=0.8
        ))
        
        fig3.update_layout(
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=15, color='#000000', family='Arial'),
            xaxis=dict(
                title=dict(text="Valor de Ventas (USD)", font=dict(size=16, color='#000000', family='Arial Black')),
                showgrid=True, 
                gridcolor='#e0e0e0',
                color='#000000',
                tickfont=dict(color='#000000', size=14, family='Arial')
            ),
            yaxis=dict(
                title=dict(text="Frecuencia", font=dict(size=16, color='#000000', family='Arial Black')),
                showgrid=True, 
                gridcolor='#e0e0e0',
                color='#000000',
                tickfont=dict(color='#000000', size=14, family='Arial')
            ),
            bargap=0.1
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        median_sales = df_filtered['total_sales'].median()
        mean_sales = df_filtered['total_sales'].mean()
        st.markdown(f"<p style='color: #000000; font-weight: 700; font-size: 1.1rem;'>Mediana: ${median_sales:.2f} | Media: ${mean_sales:.2f}</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Matriz de Rendimiento por Categoría")
        
        category_stats = df_filtered.groupby('category').agg({
            'orders_count': 'sum',
            'total_sales': 'sum'
        }).reset_index()
        
        fig4 = go.Figure(go.Scatter(
            x=category_stats['orders_count'],
            y=category_stats['total_sales'],
            mode='markers+text',
            marker=dict(
                size=category_stats['total_sales']/100,
                color=category_stats['total_sales'],
                colorscale=['#ffd4e5', '#ffb3d9', '#ff8fc7', '#f06aa7', '#d4568c'],
                showscale=True,
                line=dict(color='#c44e81', width=2),
                colorbar=dict(
                    title=dict(text="Ventas (USD)", font=dict(color='#000000', size=14)),
                    tickfont=dict(color='#000000', size=12)
                )
            ),
            text=category_stats['category'],
            textposition='top center',
            textfont=dict(size=13, color='#000000', family='Arial Black'),
            hovertemplate='<b>%{text}</b><br>Órdenes: %{x}<br>Ventas: $%{y:,.0f}<extra></extra>'
        ))
        
        fig4.update_layout(
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=15, color='#000000', family='Arial'),
            xaxis=dict(
                title=dict(text="Total de Órdenes", font=dict(size=16, color='#000000', family='Arial Black')),
                showgrid=True, 
                gridcolor='#e0e0e0',
                color='#000000',
                tickfont=dict(color='#000000', size=13)
            ),
            yaxis=dict(
                title=dict(text="Ventas Totales (USD)", font=dict(size=16, color='#000000', family='Arial Black')),
                showgrid=True, 
                gridcolor='#e0e0e0',
                color='#000000',
                tickfont=dict(color='#000000', size=13)
            )
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("### Análisis Complementario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Ventas por Mes del Año")
        monthly_sales = df_filtered.groupby('month')['total_sales'].sum().reset_index()
        month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        monthly_sales['month_name'] = monthly_sales['month'].apply(lambda x: month_names[x-1])
        
        fig5 = go.Figure(go.Bar(
            x=monthly_sales['month_name'],
            y=monthly_sales['total_sales'],
            marker=dict(
                color=monthly_sales['total_sales'],
                colorscale=['#ffd4e5', '#ffb3d9', '#ff8fc7', '#f06aa7', '#d4568c'],
                line=dict(color='#c44e81', width=2)
            ),
            text=[f'${val:,.0f}' for val in monthly_sales['total_sales']],
            textposition='outside',
            textfont=dict(size=14, color='#000000', family='Arial Black')
        ))
        
        fig5.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=15, color='#000000', family='Arial'),
            xaxis=dict(
                title=dict(text="Mes", font=dict(size=16, color='#000000', family='Arial Black')),
                showgrid=False, 
                color='#000000',
                tickfont=dict(color='#000000', size=14, family='Arial')
            ),
            yaxis=dict(
                title=dict(text="Ventas (USD)", font=dict(size=16, color='#000000', family='Arial Black')),
                showgrid=True, 
                gridcolor='#e0e0e0',
                color='#000000',
                tickfont=dict(color='#000000', size=14, family='Arial')
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        st.markdown("#### Top 5 Productos por Ingresos")
        top_products = df_filtered.groupby('product_id')['total_sales'].sum().sort_values(ascending=False).head(5)
        
        colors_pie = ['#d4568c', '#f06aa7', '#ff8fc7', '#ffb3d9', '#ffd4e5']
        
        fig6 = go.Figure(go.Pie(
            labels=[f"Producto {i+1}" for i in range(len(top_products))],
            values=top_products.values,
            hole=0.4,
            marker=dict(colors=colors_pie, line=dict(color='white', width=3)),
            textinfo='label+percent',
            textfont=dict(size=16, color='#000000', family='Arial Black'),
            hovertemplate='<b>%{label}</b><br>Ventas: $%{value:,.0f}<br>Porcentaje: %{percent}<extra></extra>'
        ))
        
        fig6.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=15, color='#000000', family='Arial')
        )
        
        st.plotly_chart(fig6, use_container_width=True)
    
    st.markdown('<p class="section-header">Predictor de Ventas con Inteligencia Artificial</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **Modelo:** Gradient Boosting Regressor | **Precisión:** R² = 0.803 | **Error:** RMSE = $255.40
    
    Este modelo utiliza inteligencia artificial para predecir ventas futuras basándose en características del producto y contexto histórico.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #000000; font-weight: 700;'>Ingrese los Parámetros de Predicción</h3>", unsafe_allow_html=True)
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        st.markdown("<p style='color: #000000; font-weight: 700; font-size: 1.1rem;'>Información del Producto</p>", unsafe_allow_html=True)
        pred_category = st.selectbox("Categoría:", options=sorted(df['category'].dropna().unique()))
        pred_weight = st.number_input("Peso (gramos):", min_value=0, max_value=50000, value=500, step=50)
        pred_photos = st.slider("Número de Fotos:", min_value=1, max_value=20, value=3)
    
    with pred_col2:
        st.markdown("<p style='color: #000000; font-weight: 700; font-size: 1.1rem;'>Período de Tiempo</p>", unsafe_allow_html=True)
        pred_year = st.number_input("Año:", min_value=2019, max_value=2026, value=2024)
        month_names_full = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        pred_month = st.selectbox("Mes:", options=list(range(1, 13)), format_func=lambda x: month_names_full[x-1])
    
    with pred_col3:
        st.markdown("<p style='color: #000000; font-weight: 700; font-size: 1.1rem;'>Métricas Operacionales</p>", unsafe_allow_html=True)
        pred_orders = st.number_input("Órdenes Esperadas:", min_value=1, max_value=100, value=5)
        pred_freight = st.number_input("Costo de Envío (USD):", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
    
    st.markdown("<p style='color: #000000; font-weight: 700; font-size: 1.1rem;'>Historial de Ventas</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        pred_sales_lag1 = st.number_input("Ventas Mes Anterior (USD):", min_value=0.0, value=150.0, step=10.0)
    with col2:
        pred_sales_lag2 = st.number_input("Ventas Hace 2 Meses (USD):", min_value=0.0, value=150.0, step=10.0)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("Generar Predicción de Ventas", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Calculando predicción..."):
            try:
                category_encoded = le.transform([pred_category])[0]
                date_numeric = pred_year * 12 + pred_month
                
                features = np.array([[
                    category_encoded, pred_year, pred_month, pred_orders,
                    pred_freight, pred_weight, pred_photos, date_numeric,
                    pred_sales_lag1, pred_sales_lag2
                ]])
                
                features_scaled = scaler.transform(features)
                prediction = gb_model.predict(features_scaled)[0]
                
                st.markdown("<h3 style='color: #000000; font-weight: 700;'>Resultado de la Predicción</h3>", unsafe_allow_html=True)
                
                result_col1, result_col2 = st.columns([1, 1])
                
                with result_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color: #000000;'>Ventas Predichas</h4>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='color: #c44e81;'>${prediction:,.2f}</h1>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with result_col2:
                    avg_sales_category = df[df['category'] == pred_category]['total_sales'].mean()
                    diff_pct = ((prediction - avg_sales_category) / avg_sales_category) * 100
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color: #000000;'>Promedio de Categoría</h4>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='color: #c44e81;'>${avg_sales_category:,.2f}</h1>", unsafe_allow_html=True)
                    
                    if diff_pct > 0:
                        st.markdown(f"<p style='color: #000000; font-weight: 700;'>Diferencia: +{diff_pct:.1f}% (por encima del promedio)</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='color: #000000; font-weight: 700;'>Diferencia: {diff_pct:.1f}% (por debajo del promedio)</p>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                
                revenue_per_order = prediction / pred_orders if pred_orders > 0 else 0
                profit_margin = (prediction - (pred_freight * pred_orders)) / prediction * 100 if prediction > 0 else 0
                confidence = "Alta" if abs(diff_pct) < 20 else "Moderada" if abs(diff_pct) < 50 else "Baja"
                
                analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
                with analysis_col1:
                    st.markdown(f"**Ingreso por Orden:** ${revenue_per_order:.2f}")
                with analysis_col2:
                    st.markdown(f"**Margen Bruto Estimado:** {profit_margin:.1f}%")
                with analysis_col3:
                    st.markdown(f"**Confianza de Predicción:** {confidence}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.success("Predicción generada exitosamente con el modelo de Gradient Boosting (R² = 0.803)")
                
            except Exception as e:
                st.error(f"Error al generar la predicción: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <p style='color: #c44e81; font-weight: bold; font-size: 1.2rem;'>Panel de Análisis de Ventas E-Commerce</p>
        <p style='color: #000000; font-weight: 600;'>MongoDB Atlas (Microsoft Azure) | Gradient Boosting ML | Big Data Analytics Project</p>
    </div>
    """, unsafe_allow_html=True)