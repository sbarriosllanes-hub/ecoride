
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. Título destacado
st.title('Sistema de Alerta Temprana de Churn - Eco-Ride')

# Load the preprocessor and model from the current working directory
@st.cache_resource
def load_model_and_preprocessor():
    try:
        preprocessor = joblib.load('pipeline_preproc.pkl')
        model = joblib.load('modelo_churn_random_forest.pkl')
        return preprocessor, model
    except FileNotFoundError:
        st.error("Error: 'pipeline_preproc.pkl' or 'modelo_churn_random_forest.pkl' not found in the app's current directory. Please ensure they are in the correct directory.")
        st.stop() # Stop the app if files are not found

preprocessor, model = load_model_and_preprocessor()

if preprocessor is not None and model is not None:
    st.sidebar.header('Datos del Cliente')

    # 2. Controles web interactivos para ingresar los datos de un cliente
    edad = st.sidebar.slider('Edad', min_value=18, max_value=80, value=30)
    plan = st.sidebar.selectbox('Plan', ['Básico', 'Premium', 'Elite'])
    uso_mensual_km = st.sidebar.slider('Uso Mensual Km', min_value=0.0, max_value=1000.0, value=150.0, step=0.1)
    soporte_tickets = st.sidebar.slider('Soporte Tickets', min_value=0, max_value=10, value=1)
    gasto_promedio = st.sidebar.slider('Gasto Promedio', min_value=10.0, max_value=200.0, value=50.0, step=0.1)
    region = st.sidebar.selectbox('Región', ['Norte', 'Sur', 'Centro'])

    # Simulate Dias_Antiguedad. For a real app, this might be calculated based on a registration date input.
    dias_antiguedad = st.sidebar.slider('Días Antigüedad', min_value=0, max_value=3650, value=365)

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'Edad': [edad],
        'Plan': [plan.lower()],
        'Uso_Mensual_Km': [uso_mensual_km],
        'Soporte_Tickets': [soporte_tickets],
        'Gasto_Promedio': [gasto_promedio],
        'Region': [region],
        'Dias_Antiguedad': [dias_antiguedad]
    })

    st.subheader('Datos de Entrada del Cliente')
    st.write(input_data)

    if st.sidebar.button('Analizar Riesgo'):
        processed_input = preprocessor.transform(input_data)
        churn_prediction = model.predict(processed_input)
        churn_proba = model.predict_proba(processed_input)[:, 1]

        st.subheader('Resultado de la Predicción')
        if churn_prediction[0] == 1:
            st.markdown(f"<h3 style='color:red;'>Alto Riesgo de Cancelación (Probabilidad: {churn_proba[0]:.2f})</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:green;'>Cliente Estable (Probabilidad: {churn_proba[0]:.2f})</h3>", unsafe_allow_html=True)

        st.write("Probabilidad de Churn:", f"{churn_proba[0]:.2f}")
        st.write("Probabilidad de No Churn:", f"{1 - churn_proba[0]:.2f}")
