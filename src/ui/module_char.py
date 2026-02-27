import streamlit as st
import pandas as pd
from src.core.montecarlo import characterize_data

def render_characterization_module():
    st.markdown("### Módulo Soporte: Caracterización de Datos Completos")
    st.markdown("Cargue un dataset histórico para encontrar la distribución de mejor ajuste usando la prueba de Anderson-Darling / KS.")
    
    uploaded_file = st.file_uploader("Cargar Datos (CSV o Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.dataframe(df.head())
            
            column = st.selectbox("Seleccione la variable a caracterizar", df.columns)
            
            if st.button("Ejecutar Caracterización"):
                data = df[column].dropna().values
                
                if len(data) < 15:
                    st.warning("Se recomienda un mínimo de 15 datos para una prueba de bondad de ajuste confiable.")
                    
                with st.spinner("Caracterizando datos..."):
                    best_dist, params = characterize_data(data)
                    
                    st.success(f"Mejor ajuste encontrado: **{best_dist.upper()}**")
                    st.write("Parámetros de la distribución:", params)
                    
                    # Store in session state for potential usage
                    if 'char_results' not in st.session_state:
                         st.session_state['char_results'] = {}
                    st.session_state['char_results'][column] = {
                        'dist': best_dist,
                        'params': params
                    }
                    
                    st.info("Los parámetros han sido calculados. Puede usar estos valores manualmente en los selectores de los otros módulos.")
                    
        except Exception as e:
            st.error(f"Error procesando el archivo: {str(e)}")
    else:
        st.info("Por favor cargue un archivo CSV o Excel con una o múltiples columnas de datos.")
