import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.core.nodal import generate_stochastic_vlp, find_intersection
from src.core.montecarlo import generate_montecarlo
from src.ui.components import st_distribution_input

def render_nodal_module(fluid_type, ipr_calc_results, iterations, system):
    st.markdown("### Módulo III: Análisis Nodal Probabilístico")
    st.markdown("Cálculo de la intersección entre la curva de Afluencia (IPR) y la curva de Levantamiento Vertical (TP/VLP).")
    
    if ipr_calc_results is None:
        st.warning("⚠️ Debes correr la Simulación IPR en el Módulo I primero.")
        return

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-weight: bold; color: #2d3748; font-size: 16px;">Variables Estocásticas VLP</div>
        </div>
        <hr style="margin-top: 10px; margin-bottom: 20px;">
        """, unsafe_allow_html=True)
        
        inputs_vlp = {}
        with st.expander("Parámetros de Tubería y Superficie", expanded=True):
            inputs_vlp['p_wh'] = st_distribution_input("Presión de Cabezal Pwh (psi)", 250.0, "pwh")
            inputs_vlp['wc'] = st_distribution_input("Corte de Agua (%)", 10.0, "wc")
            inputs_vlp['md_total'] = st.number_input("Profundidad Medida (ft)", min_value=100.0, value=5000.0, step=100.0)
            inputs_vlp['D'] = st_distribution_input("Diámetro Interno (in)", 2.992, "dia")
            inputs_vlp['roughness'] = st_distribution_input("Rugosidad Absoluta (in)", 0.0006, "rough")
            
        run_sim = st.button("▶ Ejecutar Análisis Nodal", use_container_width=True, type="primary")

    with col2:
        st.markdown("""
        <div style="font-weight: bold; color: #2d3748; font-size: 18px;">Gráfica IPR - TP Probabilística</div>
        <div style="color: #718096; font-size: 13px; margin-bottom: 20px;">Intersección simulada con Monte Carlo</div>
        """, unsafe_allow_html=True)
        
        if run_sim:
            with st.spinner("Calculando iteraciones Monte Carlo para VLP y analizando intersección..."):
                try:
                    # Generate VLP Distributions
                    vecs_vlp = {}
                    for k_name in ['p_wh', 'wc', 'D', 'roughness']:
                        d_type, d_params = inputs_vlp[k_name]
                        min_lim, max_lim = 0.001, None
                        if k_name == 'wc':
                            min_lim, max_lim = 0.0, 100.0
                        
                        user_min = d_params.pop('min_limit', None)
                        user_max = d_params.pop('max_limit', None)
                        if user_min is not None: min_lim = user_min
                        if user_max is not None: max_lim = user_max
                        
                        v = generate_montecarlo(iterations, d_type, d_params, min_limit=min_lim, max_limit=max_lim)
                        if k_name == 'wc': v = v / 100.0
                        vecs_vlp[k_name] = v

                    # Recover IPR Results
                    qi_array = ipr_calc_results['q_sim']
                    unidad_q = ipr_calc_results['unidad_q']
                    
                    # Para generar la curva, evaluamos en N puntos de Q
                    max_q = np.percentile(qi_array, 95) * 1.5 # Hasta 1.5x del P5 estimado
                    q_eval = np.linspace(0, max_q, 15)
                    
                    # Generar matriz estocástica VLP (shape: 15, iterations)
                    vlp_matrix = generate_stochastic_vlp(
                        rates=q_eval,
                        p_wh_dist=vecs_vlp['p_wh'],
                        wc_dist=vecs_vlp['wc'],
                        roughness_dist=vecs_vlp['roughness'],
                        d_dist=vecs_vlp['D'],
                        md_total=inputs_vlp['md_total'],
                        iterations=iterations
                    )
                    
                    # Generar matriz estocástica IPR (aproximación cuadrática simplificada como Vogel para demo)
                    # En la realidad, reutilizaríamos el motor IPR del modulo 1 iterativo. Aquí aproximamos usando Pwf simulada inversa:
                    
                    # Haremos una aproximación: p_ws = pr_media
                    pr_avg = 3000.0 # Valor default si no se tiene
                    if 'pr' in ipr_calc_results['vecs']:
                        pr_avg = np.mean(ipr_calc_results['vecs']['pr'])
                    
                    ipr_matrix = np.zeros((len(q_eval), iterations))
                    for j in range(iterations):
                        q_max = qi_array[j]
                        for i, q in enumerate(q_eval):
                            if q >= q_max:
                                ipr_matrix[i, j] = 0.0
                            else:
                                # Simplificacion tipo Vogel: q / q_max = 1 - 0.2(pwf/pr) - 0.8(pwf/pr)^2
                                # pwf/pr = (sqrt(81 - 80(Q/Qmax)) - 1) / 8
                                val = (np.sqrt(81 - 80*(q/q_max)) - 1) / 8
                                ipr_matrix[i, j] = val * pr_avg
                    
                    # Intersection
                    q_eq, pwf_eq = find_intersection(q_eval, ipr_matrix, vlp_matrix)
                    
                    # Filter non-flowing
                    valid_flow = q_eq > 0
                    q_eq_valid = q_eq[valid_flow]
                    
                    if len(q_eq_valid) == 0:
                        st.error("El pozo no fluye bajo estas condiciones (La VLP siempre está por encíma de la IPR).")
                        return
                    
                    q_p90 = np.percentile(q_eq_valid, 10)
                    q_p50 = np.percentile(q_eq_valid, 50)
                    q_p10 = np.percentile(q_eq_valid, 90)
                    q_mean = np.mean(q_eq_valid)
                    
                    # Plots
                    fig = go.Figure()
                    
                    # Plot P50 IPR and VLP
                    ipr_p50 = np.percentile(ipr_matrix, 50, axis=1)
                    vlp_p50 = np.percentile(vlp_matrix, 50, axis=1)
                    ipr_p10 = np.percentile(ipr_matrix, 90, axis=1)
                    ipr_p90 = np.percentile(ipr_matrix, 10, axis=1)
                    vlp_p10 = np.percentile(vlp_matrix, 10, axis=1)
                    vlp_p90 = np.percentile(vlp_matrix, 90, axis=1)
                    
                    fig.add_trace(go.Scatter(x=q_eval, y=ipr_p50, mode='lines', line=dict(color='blue', width=3), name='IPR P50'))
                    fig.add_trace(go.Scatter(x=q_eval, y=vlp_p50, mode='lines', line=dict(color='red', width=3), name='TP/VLP P50'))
                    
                    # Bandas (Fill) para IPR
                    fig.add_trace(go.Scatter(x=np.concatenate([q_eval, q_eval[::-1]]), 
                                             y=np.concatenate([ipr_p10, ipr_p90[::-1]]), 
                                             fill='toself', fillcolor='rgba(0,0,255,0.1)', line=dict(color='rgba(255,255,255,0)'), 
                                             name='Banda IPR (P10-P90)'))
                                             
                    # Bandas (Fill) para VLP
                    fig.add_trace(go.Scatter(x=np.concatenate([q_eval, q_eval[::-1]]), 
                                             y=np.concatenate([vlp_p10, vlp_p90[::-1]]), 
                                             fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'), 
                                             name='Banda TP (P10-P90)'))
                                             
                    # Nube de puntos de intersección
                    # Para no sobrecargar el gráfico, ploteamos una muestra aleatoria de 500 puntos si N es grande
                    sample_size = min(len(q_eq_valid), 500)
                    idx_sample = np.random.choice(len(q_eq_valid), sample_size, replace=False)
                    fig.add_trace(go.Scatter(x=q_eq_valid[idx_sample], y=pwf_eq[valid_flow][idx_sample], mode='markers', 
                                             marker=dict(size=4, color='green', opacity=0.3), name='Intersección Estocástica'))
                    
                    fig.update_layout(
                        title="Análisis Nodal Estocástico (IPR vs TP)",
                        xaxis_title=f"Gasto ({unidad_q})",
                        yaxis_title="Presión de Fondo Pwf (psi)",
                        plot_bgcolor='white',
                        height=500,
                        xaxis=dict(showgrid=True, gridcolor='#edf2f7'),
                        yaxis=dict(showgrid=True, gridcolor='#edf2f7')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stats
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0;">
                            <div style="color: #718096; font-size: 11px; font-weight: bold; margin-bottom: 5px;">GASTO OPERATIVO P50</div>
                            <div style="color: #2b6cb0; font-size: 24px; font-weight: bold;">{int(q_p50):,} <span style="font-size: 14px;">{unidad_q}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        prob_flow = (len(q_eq_valid) / iterations) * 100
                        st.markdown(f"""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0;">
                            <div style="color: #718096; font-size: 11px; font-weight: bold; margin-bottom: 5px;">PROBABILIDAD DE FLUJO</div>
                            <div style="color: {'#38a169' if prob_flow > 90 else '#d69e2e'}; font-size: 24px; font-weight: bold;">{prob_flow:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0;">
                            <div style="color: #718096; font-size: 11px; font-weight: bold; margin-bottom: 5px;">RANGO (P10-P90)</div>
                            <div style="color: #2b6cb0; font-size: 24px; font-weight: bold;">{int(q_p10 - q_p90):,} <span style="font-size: 14px;">{unidad_q}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Tornado Chart (Spearman)
                    corrs = {}
                    import scipy.stats as scipy_stats
                    
                    # Merge IPR and VLP vecs for sensitivity
                    all_vecs = {**ipr_calc_results['vecs'], **vecs_vlp}
                    
                    for k_name, v_array in all_vecs.items():
                        if isinstance(v_array, np.ndarray) and len(v_array) > 1 and np.std(v_array) > 0:
                            # Use full array for correlation, masking invalid flow with 0 or drop them
                            # Dropping non flowing iterations for correlation
                            corr, _ = scipy_stats.spearmanr(v_array[valid_flow], q_eq_valid)
                            if not np.isnan(corr):
                                corrs[k_name] = corr
                                
                    if corrs:
                        st.markdown("<hr style='margin: 30px 0px;'>", unsafe_allow_html=True)
                        st.markdown("""
                        <div style="font-weight: bold; color: #2d3748; font-size: 18px;">Análisis de Sensibilidad (Tornado)</div>
                        <div style="color: #718096; font-size: 13px; margin-bottom: 20px;">Impacto de las variables IPR y VLP en el Gasto Operativo</div>
                        """, unsafe_allow_html=True)
                        
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
                            xaxis=dict(showgrid=True, gridcolor='#edf2f7', range=[-1.1, 1.1], zeroline=True, zerolinecolor='black'),
                            height=300
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                except Exception as e:
                    import traceback
                    st.error(f"Error procesando el Análisis Nodal: {str(e)} \n {traceback.format_exc()}")
        else:
            st.info("Ingresa los parámetros de la tubería y presiona Ejecutar Análisis Nodal.")
