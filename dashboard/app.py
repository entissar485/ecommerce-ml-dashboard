cat > app.py << 'EOF'
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
import os
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Configuración de la página
st.set_page_config(
    page_title="E-commerce Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Conexión a MongoDB
@st.cache_resource
def get_mongo_client():
    mongo_uri = st.secrets.get("MONGO_URI", os.getenv("MONGO_URI"))
    db_name = st.secrets.get("DB_NAME", os.getenv("DB_NAME", "ecommerce_db"))
    client = MongoClient(mongo_uri)
    return client[db_name]

# Cargar datos
@st.cache_data(ttl=600)
def load_data():
    db = get_mongo_client()
    
    # Cargar colecciones
    products_df = pd.DataFrame(list(db.products.find()))
    customers_df = pd.DataFrame(list(db.customers.find()))
    orders_df = pd.DataFrame(list(db.orders.find()))
    
    # Limpiar _id de MongoDB
    for df in [products_df, customers_df, orders_df]:
        if '_id' in df.columns:
            df.drop('_id', axis=1, inplace=True)
    
    return products_df, customers_df, orders_df

# Entrenar modelo simple
@st.cache_resource
def train_simple_model():
    db = get_mongo_client()
    orders_df = pd.DataFrame(list(db.orders.find()))
    
    if len(orders_df) == 0:
        return None, None, None
    
    # Preparar datos
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    orders_df['month'] = orders_df['order_date'].dt.month
    orders_df['day_of_week'] = orders_df['order_date'].dt.dayofweek
    
    # Features para el modelo
    features = ['quantity', 'month', 'day_of_week']
    X = orders_df[features]
    y = orders_df['total_price']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Calcular métricas
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return model, train_score, test_score

# Título principal
st.title("🛍️ E-commerce Analytics Dashboard")
st.markdown("---")

# Cargar datos
try:
    products_df, customers_df, orders_df = load_data()
    
    # Sidebar - Filtros
    st.sidebar.header("📊 Filtros")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Productos", len(products_df))
    
    with col2:
        st.metric("Total Clientes", len(customers_df))
    
    with col3:
        st.metric("Total Órdenes", len(orders_df))
    
    with col4:
        total_revenue = orders_df['total_price'].sum()
        st.metric("Ingresos Totales", f"${total_revenue:,.2f}")
    
    st.markdown("---")
    
    # Análisis de Productos
    st.header("📦 Análisis de Productos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de precios
        fig_price = px.histogram(
            products_df, 
            x='price',
            title='Distribución de Precios de Productos',
            labels={'price': 'Precio ($)', 'count': 'Cantidad'},
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Stock por categoría
        stock_by_category = products_df.groupby('category')['stock'].sum().reset_index()
        fig_stock = px.bar(
            stock_by_category,
            x='category',
            y='stock',
            title='Stock Total por Categoría',
            labels={'stock': 'Stock Total', 'category': 'Categoría'},
            color='stock',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_stock, use_container_width=True)
    
    # Análisis de Ventas
    st.header("💰 Análisis de Ventas")
    
    # Preparar datos temporales
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    daily_sales = orders_df.groupby(orders_df['order_date'].dt.date).agg({
        'total_price': 'sum',
        'order_id': 'count'
    }).reset_index()
    daily_sales.columns = ['date', 'revenue', 'orders']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ventas diarias
        fig_daily = px.line(
            daily_sales,
            x='date',
            y='revenue',
            title='Ingresos Diarios',
            labels={'date': 'Fecha', 'revenue': 'Ingresos ($)'},
            markers=True
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with col2:
        # Órdenes diarias
        fig_orders = px.line(
            daily_sales,
            x='date',
            y='orders',
            title='Número de Órdenes Diarias',
            labels={'date': 'Fecha', 'orders': 'Órdenes'},
            markers=True,
            color_discrete_sequence=['#ff7f0e']
        )
        st.plotly_chart(fig_orders, use_container_width=True)
    
    # Modelo Predictivo
    st.header("🤖 Modelo Predictivo")
    
    with st.spinner('Entrenando modelo...'):
        model, train_score, test_score = train_simple_model()
    
    if model is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Precisión Entrenamiento", f"{train_score:.2%}")
        
        with col2:
            st.metric("Precisión Prueba", f"{test_score:.2%}")
        
        with col3:
            st.metric("Tipo de Modelo", "Random Forest")
        
        st.success("✅ Modelo entrenado exitosamente")
        
        # Predictor interactivo
        st.subheader("🔮 Predictor de Ventas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quantity = st.number_input("Cantidad", min_value=1, max_value=100, value=1)
        
        with col2:
            month = st.selectbox("Mes", range(1, 13), index=0)
        
        with col3:
            day_of_week = st.selectbox("Día de la semana", 
                                       ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'],
                                       index=0)
        
        day_mapping = {'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3, 
                      'Viernes': 4, 'Sábado': 5, 'Domingo': 6}
        
        if st.button("🎯 Predecir Venta"):
            prediction_input = [[quantity, month, day_mapping[day_of_week]]]
            prediction = model.predict(prediction_input)[0]
            
            st.success(f"💵 Predicción de venta: ${prediction:,.2f}")
    else:
        st.warning("⚠️ No hay suficientes datos para entrenar el modelo")
    
    # Análisis de Clientes
    st.header("👥 Análisis de Clientes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución por país
        country_dist = customers_df['country'].value_counts().reset_index()
        country_dist.columns = ['country', 'count']
        fig_country = px.pie(
            country_dist,
            values='count',
            names='country',
            title='Distribución de Clientes por País'
        )
        st.plotly_chart(fig_country, use_container_width=True)
    
    with col2:
        # Top clientes por gastos
        customer_spending = orders_df.groupby('customer_id')['total_price'].sum().reset_index()
        customer_spending = customer_spending.nlargest(10, 'total_price')
        
        fig_top_customers = px.bar(
            customer_spending,
            x='customer_id',
            y='total_price',
            title='Top 10 Clientes por Gasto Total',
            labels={'customer_id': 'ID Cliente', 'total_price': 'Gasto Total ($)'},
            color='total_price',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_top_customers, use_container_width=True)
    
    st.markdown("---")
    st.caption("Dashboard actualizado en tiempo real desde MongoDB Atlas")
    
except Exception as e:
    st.error(f"❌ Error al cargar datos: {str(e)}")
    st.info("Verifica que las credenciales de MongoDB estén configuradas correctamente en Secrets.")
EOF