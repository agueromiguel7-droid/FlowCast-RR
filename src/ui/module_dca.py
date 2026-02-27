import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from src.core.montecarlo import generate_montecarlo
from src.core.models_dca import generar_perfil_montecarlo

def calculate_spearman_correlation(inputs_dict, output_vec):
    """Calcula la correlación de Spearman para el Tornado Plot"""
    correlations = {}
    for name, vec in inputs_dict.items():
        corr, _ = stats.spearmanr(vec, output_vec)
        # Handle constant inputs
        if np.isnan(corr):
            corr = 0.0
        correlations[name] = corr
    return correlations

def render_dca_module(fluido, iteraciones):
    st.markdown("### Módulo II: Pronóstico (DCA)")
    
    # KPIs Top Row
    kpi1, kpi2, kpi3 = st.columns(3)
    
    # Left column: Perfil
    # Right column: Tornado
    col_perfil, col_tornado = st.columns([2.5, 1])
    
    with col_perfil:
        st.markdown("<div style='font-weight: bold; font-size: 16px; margin-bottom: 10px;'>Perfil de Producción Probabilístico</div>", unsafe_allow_html=True)
        # Controles
        q_abandono = st.number_input("Límite Económico (q_abandono, bbl/d)", value=20.0, step=5.0)
        
        # Simulación de variables para DCA
        # Asumiendo inputs default para la demostración
        t_years = 20
        t_steps = np.arange(1, t_years * 12 + 1) # Meses
        
        # Generar inputs probabilísticos
        qi_sim = generate_montecarlo(iteraciones, 'normal', {'mu': 1240, 'sigma': 124})
        D_sim = generate_montecarlo(iteraciones, 'uniforme', {'min': 0.05, 'max': 0.15}) / 12.0 # tasa mensual
        b_sim = generate_montecarlo(iteraciones, 'uniforme', {'min': 0.3, 'max': 0.8})
        
        q_t, eur = generar_perfil_montecarlo(qi_sim, D_sim, b_sim, t_steps, modelo="hiperbolica", q_abandono=q_abandono)
        
        # Calcular percentiles EUR
        eur_p90 = np.percentile(eur, 10) / 1000 # MBBL
        eur_p50 = np.percentile(eur, 50) / 1000
        eur_p10 = np.percentile(eur, 90) / 1000
        
        # Llenar KPIs
        with kpi1:
             st.markdown(f"""
             <div style="background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">
                 <div style="color: #4a5568; font-size: 12px; font-weight: bold; margin-bottom: 5px;">EUR P90</div>
                 <div style="color: #1a202c; font-size: 32px; font-weight: bold;">{eur_p90:.1f} <span style="font-size: 14px; color: #a0aec0;">MBBL</span></div>
                 <div style="color: #e53e3e; font-size: 12px; margin-top: 5px;">↘ Escenario conservador</div>
             </div>
             """, unsafe_allow_html=True)
        with kpi2:
             st.markdown(f"""
             <div style="background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; border-top: 4px solid #ed8936;">
                 <div style="color: #dd6b20; font-size: 12px; font-weight: bold; margin-bottom: 5px;">EUR P50 (MEDIANA)</div>
                 <div style="color: #1a202c; font-size: 32px; font-weight: bold;">{eur_p50:.1f} <span style="font-size: 14px; color: #a0aec0;">MBBL</span></div>
                 <div style="color: #38a169; font-size: 12px; margin-top: 5px;">↗ Escenario más probable</div>
             </div>
             """, unsafe_allow_html=True)
        with kpi3:
             st.markdown(f"""
             <div style="background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">
                 <div style="color: #4a5568; font-size: 12px; font-weight: bold; margin-bottom: 5px;">EUR P10</div>
                 <div style="color: #1a202c; font-size: 32px; font-weight: bold;">{eur_p10:.1f} <span style="font-size: 14px; color: #a0aec0;">MBBL</span></div>
                 <div style="color: #dd6b20; font-size: 12px; margin-top: 5px;">⚡ Escenario optimista</div>
             </div>
             """, unsafe_allow_html=True)
             
        # Calcular percentiles de perfiles temporales
        q_t_p90 = np.percentile(q_t, 10, axis=0)
        q_t_p50 = np.percentile(q_t, 50, axis=0)
        q_t_p10 = np.percentile(q_t, 90, axis=0)
        q_t_mean = np.mean(q_t, axis=0)
        
        # Plotly Line Chart
        fig = go.Figure()
        
        # Area sombreada P90-P10
        x_area = np.concatenate([t_steps, t_steps[::-1]])
        y_area = np.concatenate([q_t_p10, q_t_p90[::-1]])
        
        fig.add_trace(go.Scatter(
            x=x_area, y=y_area,
            fill='toself',
            fillcolor='rgba(237, 137, 54, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Cono de Incertidumbre',
            showlegend=False
        ))
        
        # Líneas
        fig.add_trace(go.Scatter(x=t_steps, y=q_t_p90, line=dict(color='#cbd5e0', width=2, dash='dash'), name='P90'))
        fig.add_trace(go.Scatter(x=t_steps, y=q_t_p50, line=dict(color='#dd6b20', width=4), name='P50'))
        fig.add_trace(go.Scatter(x=t_steps, y=q_t_p10, line=dict(color='#cbd5e0', width=2, dash='dash'), name='P10'))
        fig.add_trace(go.Scatter(x=t_steps, y=q_t_mean, line=dict(color='#3182ce', width=2), name='Media'))
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor='white',
            xaxis_title='Tiempo (Meses)',
            yaxis_title='Gasto (BBL/D)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_type="log" # Logarítmico por defecto, normalmente útil en DCA
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_tornado:
        st.markdown("<div style='font-weight: bold; font-size: 16px; margin-bottom: 20px;'>Sensibilidad<br>(Spearman)</div>", unsafe_allow_html=True)
        
        corrs = calculate_spearman_correlation({
            "Gasto Inicial (qi)": qi_sim,
            "Tasa (D)": D_sim,
            "Exponente (b)": b_sim
        }, eur)
        
        # Ordenar e graficar
        sorted_corrs = dict(sorted(corrs.items(), key=lambda item: abs(item[1])))
        labels = list(sorted_corrs.keys())
        values = list(sorted_corrs.values())
        
        colors = ['#e53e3e' if v < 0 else '#3182ce' for v in values]
        
        fig2 = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition='outside'
        ))
        fig2.update_layout(
            margin=dict(l=10, r=30, t=10, b=10),
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#edf2f7', range=[-1, 1], zeroline=True, zerolinecolor='black'),
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # 6.2 Implementar exportación de reporte Excel consolidado
        import pandas as pd
        import io
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight: bold; font-size: 14px; margin-bottom: 10px;'>Exportar Resultados</div>", unsafe_allow_html=True)
        
        df_export = pd.DataFrame({
            "Mes": t_steps,
            "P90 (bbl/d)": q_t_p90,
            "P50 (bbl/d)": q_t_p50,
            "P10 (bbl/d)": q_t_p10,
            "Media (bbl/d)": q_t_mean
        })
        
        # Guardar en memoria
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Perfil Producción')
            # Additional sheet for KPIs
            df_kpi = pd.DataFrame({"Métrica": ["EUR P90", "EUR P50", "EUR P10"], "Valor (MBBL)": [eur_p90, eur_p50, eur_p10]})
            df_kpi.to_excel(writer, index=False, sheet_name='Resumen Directivo')
        output.seek(0)
        
        st.download_button(
            label="Generar Reporte de Decisión (Excel)",
            data=output,
            file_name="Reporte_FlowCast_DCA.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
