import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.core.stats import fit_all_distributions, DISTRIBUTIONS

def render_characterization_module():
    st.markdown("### Módulo 1: Caracterización de Datos Completos")
    st.markdown("Basado en el motor analítico de QuantX Pro para identificación probabilística de variables.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### Entrada de Datos")
        with st.container(border=True):
            data_input = st.text_area("Pegar datos numéricos:", height=300, help="Copie y pegue sus datos numéricos de la variable a analizar.")
            uploaded_file = st.file_uploader("O cargar archivo CSV/Excel", type=["csv", "xlsx"])
            run_btn = st.button("Ejecutar Análisis", type="primary", use_container_width=True)
            
            if st.button("Limpiar Memoria", use_container_width=True):
                if 'dist_results' in st.session_state:
                    del st.session_state['dist_results']
                st.rerun()

    with col2:
        if run_btn and (data_input or uploaded_file):
            with st.spinner("Analizando distribuciones vs Anderson-Darling..."):
                try:
                    if uploaded_file:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        # Busca la primera columna numérica
                        data = df.select_dtypes(include=np.number).iloc[:, 0].dropna().values
                    else:
                        data = np.array([float(x) for x in data_input.replace(',', '.').split() if x.strip()])
                    
                    if len(data) < 3:
                        st.error("Se requieren al menos 3 puntos de datos numéricos para procesar.")
                    else:
                        results = fit_all_distributions(data)
                        st.session_state['dist_results'] = results
                        st.session_state['data_vector'] = data
                except Exception as e:
                    st.error(f"Error procesando los datos de entrada: {str(e)}")

        if 'dist_results' in st.session_state:
            results = st.session_state['dist_results']
            data = st.session_state['data_vector']
            
            st.markdown("### Resultados del Ajuste")
            
            dist_list = results['Distribution'].tolist()
            selected_dist_name = st.selectbox("Seleccionar Distribución para Análisis Estocástico", dist_list, index=0)
            
            selected_row = results[results['Distribution'] == selected_dist_name].iloc[0]
            selected_params = selected_row['_params_obj']
            
            st.markdown(f"**Estadísticos Muestrales:** Media={np.mean(data):.4f} | Desv. Est.={np.std(data):.4f} | n={len(data)}")
            st.markdown("<hr style='margin-top: 5px; margin-bottom: 15px;'>", unsafe_allow_html=True)
            
            def compact_metric(label, value):
                st.markdown(f"<div style='margin-bottom:5px; color: #555; font-size:12px;'>{label}</div><div style='font-weight:bold; font-size:16px; color:#2b6cb0;'>{value}</div>", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1: compact_metric("Distribución", selected_dist_name)
            with m2: compact_metric("Anderson-Darling St.", f"{selected_row['AD Statistic']:.4f}")
            with m3: compact_metric("P-Value (K-S)", f"{selected_row['P-Value']:.4f}")
            
            st.markdown("<br><div style='font-weight:bold; color:#4a5568;'>Parámetros Estimados:</div>", unsafe_allow_html=True)
            
            param_metrics = []
            if selected_dist_name == "Normal":
                loc, scale = selected_params
                param_metrics = [("Media (µ)", loc), ("Desv. Est. (σ)", scale)]
            elif selected_dist_name == "Lognormal (2P)":
                s, loc, scale = selected_params
                param_metrics = [("Mediana", scale), ("Log. Desv. Est.", s)]
            elif selected_dist_name == "Weibull (2P)":
                c, loc, scale = selected_params
                param_metrics = [("Forma (k)", c), ("Escala (lambda)", scale)]
            elif selected_dist_name == "Gamma (2P)":
                a, loc, scale = selected_params
                param_metrics = [("Forma (Alpha/k)", a), ("Escala (Theta)", scale)]
            elif selected_dist_name == "Triangular":
                c, loc, scale = selected_params
                param_metrics = [("Mínimo", loc), ("Moda", loc + c*scale), ("Máximo", loc+scale)]
            elif selected_dist_name == "Beta":
                a, b, loc, scale = selected_params
                param_metrics = [("Alpha", a), ("Beta", b), ("Min", loc), ("Max", loc+scale)]
            elif selected_dist_name == "Exponential (1P)":
                loc, scale = selected_params
                param_metrics = [("Tasa (Lambda)", 1.0/scale), ("Media", scale)]
            
            p_cols = st.columns(len(param_metrics) if len(param_metrics) > 0 else 1)
            for i, (label, val) in enumerate(param_metrics):
                 with p_cols[i]:
                      st.markdown(f"<div style='font-size:13px; margin-top:10px;'>{label}: <b>{val:.4f}</b></div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            tab_comp, tab_viz = st.tabs(["Comparativa de Modelos", "Visualización Avanzada"])
            
            with tab_comp:
                st.markdown("#### Jerarquía de Distribuciones (Ranking)")
                
                def highlight_best(s):
                    is_best = s.name == results.index[0]
                    return ['background-color: #d1fae5; color: black' if is_best else '' for _ in s]
                
                def color_pvalue(val):
                    if pd.isna(val): return ''
                    if val > 0.05: return 'background-color: #86efac; color: black' 
                    elif val > 0.01: return 'background-color: #fde047; color: black' 
                    else: return 'background-color: #fca5a5; color: black' 

                display_df = results.drop(columns=['_params_obj'], errors='ignore')
                st.dataframe(
                    display_df.style.apply(highlight_best, axis=1).map(color_pvalue, subset=['P-Value']).format({'P-Value': '{:.4f}', 'AD Statistic': '{:.4f}'}),
                    use_container_width=True, hide_index=True
                )
                
            with tab_viz:
                plot_type = st.radio("Tipo de Gráfico", ["Densidad (PDF)", "Acumulada Directa (CDF)", "Probability Plot (Q-Q)"], horizontal=True)
                fig = go.Figure()
                
                x_range = np.linspace(min(data), max(data), 200)
                dist_func = DISTRIBUTIONS.get(selected_dist_name)
                
                if plot_type == "Densidad (PDF)":
                    fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Datos Históricos', opacity=0.5, marker_color='#a0aec0'))
                    if dist_func:
                        y = dist_func.pdf(x_range, *selected_params)
                        fig.add_trace(go.Scatter(x=x_range, y=y, mode='lines', name=f"{selected_dist_name}", line=dict(color='#2b6cb0', width=3)))
                        fig.update_layout(yaxis_title="Densidad")

                elif plot_type == "Acumulada Directa (CDF)":
                     sorted_data = np.sort(data)
                     y_data = np.arange(1, len(data) + 1) / len(data)
                     fig.add_trace(go.Scatter(x=sorted_data, y=y_data, mode='markers', name='Frecuencia Empírica', marker=dict(color='#4a5568', size=6)))
                     if dist_func:
                        y = dist_func.cdf(x_range, *selected_params)
                        fig.add_trace(go.Scatter(x=x_range, y=y, mode='lines', name=f"{selected_dist_name}", line=dict(color='#2b6cb0', width=3)))
                        fig.update_layout(yaxis_title="Probabilidad Acumulada")
                        
                elif plot_type == "Probability Plot (Q-Q)":
                    if dist_func:
                        n = len(data)
                        sorted_data = np.sort(data)
                        p_val = (np.arange(1, n+1) - 0.5) / n
                        theoretical_quantiles = dist_func.ppf(p_val, *selected_params)
                        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name='Datos Muestrales', marker=dict(color='#2b6cb0', size=7)))
                        
                        min_val = min(min(theoretical_quantiles), min(sorted_data))
                        max_val = max(max(theoretical_quantiles), max(sorted_data))
                        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ajuste 1:1', line=dict(color='#e53e3e', dash='dash')))
                        
                        fig.update_layout(xaxis_title=f"Desviación Teórica ({selected_dist_name})", yaxis_title="Desviación Real (Muestra)")

                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='white', title=f"Estudio Predictivo: {selected_dist_name}", height=450, xaxis=dict(showgrid=True, gridcolor='#edf2f7'), yaxis=dict(showgrid=True, gridcolor='#edf2f7'))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Cargue información desde el panel izquierdo (Texto Manual o Archivo Excel/CSV) para inicializar.")
