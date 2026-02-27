import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.core.montecarlo import generate_montecarlo
from src.core.models_ipr import ipr_aceite_darcy

def render_ipr_module(fluid_type, model_type, iterations, system):
    st.markdown("### Módulo I: Afluencia (IPR)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-weight: bold; color: #2d3748;">Variables de Entrada</div>
            <a href="#" style="font-size: 12px; color: #2b6cb0; text-decoration: none;">Reset Defaults</a>
        </div>
        <hr style="margin-top: 10px; margin-bottom: 20px;">
        """, unsafe_allow_html=True)
        
        with st.expander("Propiedades de Roca", expanded=True):
            r1, r2 = st.columns(2)
            perm = r1.number_input("PERMEABILIDAD (MD)", value=150.0, help="Permeabilidad absoluta de la roca")
            poro = r2.number_input("POROSIDAD (%)", value=18.0, help="Porosidad efectiva")
            espesor = st.slider("ESPESOR NETO (FT)", min_value=10.0, max_value=100.0, value=50.0)
            
        with st.expander("Propiedades de Fluido", expanded=True):
            viscosidad = st.number_input("VISCOSIDAD (CP)", value=1.2)
            bo = st.number_input("FACTOR VOLUMÉTRICO (BO)", value=1.05)
            
        with st.expander("Presiones y Otros", expanded=True):
            pr = st.number_input("Presión de Yacimiento (psi)", value=3000.0)
            pwf = st.number_input("Presión de Fondo Fluyente (psi)", value=2000.0)
            re = st.number_input("Radio de Drenaje (ft)", value=1500.0)
            rw = st.number_input("Radio de Pozo (ft)", value=0.328)
            skin = st.number_input("Daño (Skin)", value=0.0)
            
        run_sim = st.button("▶ Run Simulation", use_container_width=True, type="primary")

    with col2:
        st.markdown("""
        <div style="font-weight: bold; color: #2d3748; font-size: 18px;">Distribución de Probabilidad (IPR)</div>
        <div style="color: #718096; font-size: 13px; margin-bottom: 20px;">Resultados estocásticos basados en simulación Monte Carlo</div>
        """, unsafe_allow_html=True)
        
        # Simulación
        if run_sim:
            with st.spinner("Calculando iteraciones..."):
                # Para MVP simplificado: asume Distribución Triangular +/- 20% para el uncertainty mode
                # Permeabilidad
                k_sim = generate_montecarlo(iterations, 'triangular', {'min': perm*0.8, 'most_likely': perm, 'max': perm*1.2})
                # Espesor
                h_sim = generate_montecarlo(iterations, 'triangular', {'min': espesor*0.9, 'most_likely': espesor, 'max': espesor*1.1})
                # Fluido
                visc_sim = generate_montecarlo(iterations, 'deterministico', {'value': viscosidad})
                bo_sim = generate_montecarlo(iterations, 'deterministico', {'value': bo})
                pr_sim = generate_montecarlo(iterations, 'normal', {'mu': pr, 'sigma': pr*0.05})
                pwf_sim = generate_montecarlo(iterations, 'deterministico', {'value': pwf})
                
                # Calcular Q
                q_sim = ipr_aceite_darcy(k_sim, h_sim, pr_sim, pwf_sim, bo_sim, visc_sim, re, rw, skin, system)
                
                # Estadísticas
                p90 = np.percentile(q_sim, 10) # 90% probabilidad de exceder
                p50 = np.percentile(q_sim, 50)
                p10 = np.percentile(q_sim, 90)
                mean_q = np.mean(q_sim)
                
                # Plotly Histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=q_sim, 
                    histnorm='probability',
                    marker_color='#cbd5e0',
                    opacity=0.75,
                    name='Frecuencia'
                ))
                
                # Líneas P90, P50, P10
                max_y = 0.1 # proxy
                fig.add_vline(x=p90, line_dash="solid", line_color="#e53e3e", annotation_text=f"P90<br>{int(p90)}", annotation_position="top left", annotation_font_color="#e53e3e")
                fig.add_vline(x=p50, line_dash="solid", line_color="#3182ce", annotation_text=f"P50<br>{int(p50)}", annotation_position="top left", annotation_font_color="#3182ce")
                fig.add_vline(x=p10, line_dash="solid", line_color="#38a169", annotation_text=f"P10<br>{int(p10)}", annotation_position="top left", annotation_font_color="#38a169")
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    plot_bgcolor='white',
                    xaxis_title='Gasto Inicial (bbl/d)',
                    yaxis_title='Frecuencia (%)',
                    showlegend=False,
                    height=500,
                    xaxis=dict(showgrid=True, gridcolor='#edf2f7'),
                    yaxis=dict(showgrid=True, gridcolor='#edf2f7', showticklabels=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tarjetas inferiores
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div style="background-color: #f7fafc; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="color: #718096; font-size: 11px; font-weight: bold; margin-bottom: 5px;">OIL RATE (AVG)</div>
                        <div style="color: #2b6cb0; font-size: 24px; font-weight: bold;">{int(mean_q):,} <span style="font-size: 14px;">bbl/d</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    j_index = mean_q / (pr - pwf) if (pr - pwf) > 0 else 0
                    st.markdown(f"""
                    <div style="background-color: #f7fafc; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="color: #718096; font-size: 11px; font-weight: bold; margin-bottom: 5px;">PRODUCTIVITY INDEX</div>
                        <div style="color: #2b6cb0; font-size: 24px; font-weight: bold;">{j_index:.1f} <span style="font-size: 14px;">J</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div style="background-color: #f7fafc; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="color: #718096; font-size: 11px; font-weight: bold; margin-bottom: 5px;">CONFIDENCE RANGE</div>
                        <div style="color: #2b6cb0; font-size: 24px; font-weight: bold;">P90 - P10</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Ajuste los parámetros a la izquierda y presione 'Run Simulation'.")
