import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.core.montecarlo import generate_montecarlo
from src.core.models_ipr import (
    ipr_aceite_desviacion_historica,
    ipr_aceite_darcy,
    ipr_aceite_darcy_empirico,
    ipr_aceite_darcy_ynf,
    ipr_aceite_vogel,
    ipr_aceite_babu_odeh,
    ipr_aceite_joshi,
    ipr_gas_pseudo_estable,
    ipr_gas_economides,
    ipr_gas_joshi_horizontal,
    ipr_gas_ynf
)
from src.ui.components import st_distribution_input
from src.core.stats import fit_all_distributions, DISTRIBUTIONS

def render_ipr_module(fluid_type, model_type, iterations, system):
    st.markdown("### Módulo I: Cálculo de Producción Inicial ($q_i$)")
    st.markdown("Cálculo basado en la capacidad de aporte del pozo (IPR).")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-weight: bold; color: #2d3748; font-size: 16px;">Variables Estocásticas</div>
        </div>
        <hr style="margin-top: 10px; margin-bottom: 20px;">
        """, unsafe_allow_html=True)
        
        # Definir configuraciones de variables por modelo
        model_reqs = {
            "1. Desviación Histórica": {"Roca": [], "Fluido": [], "Presiones": ['q_det', 'fact_desv'], "Latex": r"q_i = Q_{hist} \pm ( Q_{hist} \cdot f_{desv} )"},
            "2. IPR Darcy - Método Analítico": {"Roca": ['perm', 'h'], "Fluido": ['visc', 'bo'], "Presiones": ['dp', 're', 'rw', 'skin'], "Latex": r"q_o = \frac{c \cdot K \cdot h \cdot (P_r - P_{wf})}{\mu_o \cdot B_o \cdot (\ln(r_e/r_w) - 0.75 + S)}"},
            "3. IPR Darcy - Método Empírico": {"Roca": ['j'], "Fluido": [], "Presiones": ['dp'], "Latex": r"q_o = J \cdot (P_r - P_{wf})"},
            "4. IPR Darcy Modificado (YNF) - Método Analítico": {"Roca": ['perm', 'kf', 'poro_m', 'poro_f', 'h'], "Fluido": ['visc', 'bo'], "Presiones": ['dp', 're', 'rw', 'skin'], "Latex": r"q_o = \frac{c \cdot K_{eq} \cdot h \cdot (P_r - P_{wf})}{\mu_o \cdot B_o \cdot (\ln(0.472 \cdot r_e/r_w) + S)}"},
            "5. IPR-Vogel": {"Roca": [], "Fluido": [], "Presiones": ['qtest', 'dp_test', 'pr_test', 'dp', 'pr'], "Latex": r"q_o = Q_{max} \cdot \left[ 1 - 0.2 \left( \frac{P_{wf}}{P_r} \right) - 0.8 \left( \frac{P_{wf}}{P_r} \right)^2 \right]"},
            "6. IPR Babu&Odeh (Pozo Horizontal) - Método Analítico": {"Roca": ['kx', 'kz', 'h'], "Fluido": ['visc', 'bo'], "Presiones": ['dp', 'area', 'len', 'rw'], "Latex": r"q_o = \frac{c \cdot b \cdot \sqrt{K_x K_z} \cdot (P_r - P_{wf})}{\mu_o \cdot B_o \cdot (\ln(\frac{\sqrt{A}}{r_w}) + \ln(C_H) - 0.75 + S_R)}"},
            "7. IPR Joshi (Pozo Horizontal) - Método Analítico": {"Roca": ['kh', 'kv', 'h'], "Fluido": ['visc', 'bo'], "Presiones": ['dp', 'len', 're', 'rw'], "Latex": r"q_o = \frac{c \cdot K_h \cdot h \cdot (P_r - P_{wf})}{B_o \cdot \mu_o \cdot \left[ \ln\left( \frac{a + \sqrt{a^2 - (L/2)^2}}{L/2} \right) + \frac{\beta h}{L} \ln\left( \frac{\beta h}{2r_w} \right) + S \right]}"},
            "8. Caudal en Estado Pseudo Estable": {"Roca": ['perm', 'h'], "Fluido": ['visc', 'z', 't'], "Presiones": ['dp2', 're', 'rw', 'skin'], "Latex": r"q_g = \frac{c \cdot K_g \cdot h \cdot (P_r^2 - P_{wf}^2)}{\mu_g \cdot Z \cdot T \cdot (\ln(r_e/r_w) - 0.75 + S)}"},
            "9. Gasto en Estado Pseudo Estable para Fracturamiento Hidráulico - Economides": {"Roca": ['perm', 'kf', 'wf', 'xf', 'h'], "Fluido": ['visc', 'z', 't'], "Presiones": ['dp2', 're', 'rw'], "Latex": r"q_g = \frac{c \cdot K_g \cdot h \cdot (P_r^2 - P_{wf}^2)}{\mu_g \cdot Z \cdot T \cdot (\ln(r_e/r_w) - \ln(0.5 x_f/r_w))}"},
            "10. Gasto en Estado Estable - Pozo Horizontal - Joshi": {"Roca": ['kh', 'kv', 'h'], "Fluido": ['visc', 'z', 't'], "Presiones": ['dp2', 'len', 're', 'rw', 'skin'], "Latex": r"q_g = \frac{c \cdot K_h \cdot h \cdot (P_r^2 - P_{wf}^2)}{\mu_g \cdot Z \cdot T \cdot \left[ \ln\left( \frac{a + \sqrt{a^2 - (L/2)^2}}{L/2} \right) + \frac{\beta h}{L} \ln\left( \frac{\beta h}{2r_w} \right) + S \right]}"},
            "11. Producción Commingled de arenas fracturadas": {"Roca": ['perm', 'h'], "Fluido": ['visc', 'z', 't'], "Presiones": ['dp2', 're', 'rw', 'skin'], "Latex": r"q_{g,total} = \sum q_{g,i}"},
            "12. Gasto en Estado Estable - Pozo Horizontal con Fractura - Joshi": {"Roca": ['kh', 'kv', 'h'], "Fluido": ['visc', 'z', 't'], "Presiones": ['dp2', 'len', 're', 'rw', 'skin'], "Latex": r"q_g = \frac{c \cdot K_h \cdot h \cdot (P_r^2 - P_{wf}^2)}{\mu_g \cdot Z \cdot T \cdot \left[ \ln\left( \frac{a + \sqrt{a^2 - (L/2)^2}}{L/2} \right) + \frac{\beta h}{L} \ln\left( \frac{\beta h}{2r_w} \right) + S \right]}"},
            "13. Gasto en Estado Estable para Yacimientos Naturalmente Fracturados": {"Roca": ['perm', 'kf', 'poro_m', 'poro_f', 'h'], "Fluido": ['visc', 'z', 't'], "Presiones": ['dp2', 're', 'rw', 'skin'], "Latex": r"q_g = \frac{c \cdot K_{eq} \cdot h \cdot (P_r^2 - P_{wf}^2)}{\mu_g \cdot Z \cdot T \cdot (\ln(0.472 \cdot r_e/r_w) + S)}"}
        }
        
        reqs = model_reqs.get(model_type, {"Roca": ['perm', 'h'], "Fluido": ['visc', 'bo', 'z', 't'], "Presiones": ['dp', 're', 'rw', 'skin'], "Latex": ""})
        
        if reqs["Latex"]:
            st.markdown("<div style='font-size: 13px; color: #718096; font-weight: bold;'>Modelo Matemático:</div>", unsafe_allow_html=True)
            st.latex(reqs["Latex"])
            st.markdown("<br>", unsafe_allow_html=True)
            
        inputs_data = {}
        
        if reqs["Roca"]:
            with st.expander("Propiedades de Roca", expanded=True):
                if 'perm' in reqs["Roca"]: inputs_data['perm'] = st_distribution_input("Permeabilidad Matriz (mD)", 150.0, "perm")
                if 'kf' in reqs["Roca"]: inputs_data['kf'] = st_distribution_input("Permeabilidad Fractura (mD)", 500.0, "kf")
                if 'kh' in reqs["Roca"]: inputs_data['kh'] = st_distribution_input("Permeabilidad Horiz. (mD)", 150.0, "kh")
                if 'kv' in reqs["Roca"]: inputs_data['kv'] = st_distribution_input("Permeabilidad Vert. (mD)", 50.0, "kv")
                if 'kx' in reqs["Roca"]: inputs_data['kx'] = st_distribution_input("Permeabilidad Kx (mD)", 100.0, "kx")
                if 'kz' in reqs["Roca"]: inputs_data['kz'] = st_distribution_input("Permeabilidad Kz (mD)", 50.0, "kz")
                if 'poro_m' in reqs["Roca"]: inputs_data['poro_m'] = st_distribution_input("Porosidad Matriz (%)", 15.0, "poro_m")
                if 'poro_f' in reqs["Roca"]: inputs_data['poro_f'] = st_distribution_input("Porosidad Fractura (%)", 2.0, "poro_f")
                if 'h' in reqs["Roca"]: inputs_data['h'] = st_distribution_input("Espesor Neto (ft)", 50.0, "esp")
                if 'j' in reqs["Roca"]: inputs_data['j'] = st_distribution_input("Índice J (bpd/psi)", 1.5, "jind")
                if 'wf' in reqs["Roca"]: inputs_data['wf'] = st_distribution_input("Ancho Fractura (ft)", 0.02, "wf")
                if 'xf' in reqs["Roca"]: inputs_data['xf'] = st_distribution_input("Media Long. Fractura (ft)", 200.0, "xf")
            
        if reqs["Fluido"]:
            with st.expander("Propiedades de Fluido", expanded=True):
                if 'visc' in reqs["Fluido"]: inputs_data['visc'] = st_distribution_input("Viscosidad (cp)", 1.2, "visc")
                if 'bo' in reqs["Fluido"]: inputs_data['bo'] = st_distribution_input("Factor Volumétrico Bo", 1.05, "bo")
                if 'z' in reqs["Fluido"]: inputs_data['z'] = st_distribution_input("Factor Z (Gas)", 0.85, "z")
                if 't' in reqs["Fluido"]: inputs_data['t'] = st_distribution_input("Temperatura (°R)", 600.0, "t")
            
        if reqs["Presiones"]:
            with st.expander("Presiones y Otros", expanded=True):
                if 'dp' in reqs["Presiones"]: inputs_data['dp'] = st_distribution_input("Diferencial Pr - Pwf (psi)", 1000.0, "dp")
                if 'dp2' in reqs["Presiones"]: inputs_data['dp2'] = st_distribution_input("Diferencial Pr² - Pwf² (psi²)", 5000000.0, "dp2")
                if 'pr' in reqs["Presiones"]: inputs_data['pr'] = st_distribution_input("Presión Yacimiento Pr (psi)", 3000.0, "pr")
                if 'pr_test' in reqs["Presiones"]: inputs_data['pr_test'] = st_distribution_input("Presión Yacimiento Pr (Test) [psi]", 3000.0, "pr_test")
                if 'dp_test' in reqs["Presiones"]: inputs_data['dp_test'] = st_distribution_input("Diferencial Pr - Pwf (Test) [psi]", 500.0, "dp_test")
                if 'qtest' in reqs["Presiones"]: inputs_data['qtest'] = st_distribution_input("Gasto de Prueba Qtest", 500.0, "qtest")
                if 're' in reqs["Presiones"]: inputs_data['re'] = st_distribution_input("Radio de Drenaje (ft)", 1500.0, "re")
                if 'rw' in reqs["Presiones"]: inputs_data['rw'] = st_distribution_input("Radio de Pozo (ft)", 0.328, "rw")
                if 'skin' in reqs["Presiones"]: inputs_data['skin'] = st_distribution_input("Daño (Skin)", 0.0, "skin")
                if 'len' in reqs["Presiones"]: inputs_data['L'] = st_distribution_input("Longitud L (ft)", 3000.0, "len")
                if 'area' in reqs["Presiones"]: inputs_data['area'] = st_distribution_input("Área Drenaje A (ft²)", 100000.0, "area")
                if 'q_det' in reqs["Presiones"]: inputs_data['q_det'] = st_distribution_input("Gasto Estimado Hist.", 1000.0, "q_det")
                if 'fact_desv' in reqs["Presiones"]: inputs_data['fact_desv'] = st_distribution_input("Factor Desviación (%)", 10.0, "fdesv")
                
                if "Commingled" in model_type:
                    st.info("Para Commingled use inputs promediados o consolidados.")
                
        run_sim = st.button("▶ Ejecutar Simulación IPR", use_container_width=True, type="primary")

    with col2:
        st.markdown("""
        <div style="font-weight: bold; color: #2d3748; font-size: 18px;">Distribución de Probabilidad (IPR)</div>
        <div style="color: #718096; font-size: 13px; margin-bottom: 20px;">Resultados estocásticos basados en simulación Monte Carlo</div>
        """, unsafe_allow_html=True)
        
        if run_sim:
            with st.spinner("Calculando iteraciones Monte Carlo..."):
                try:
                    # Generar vectores de variables
                    vecs = {}
                    for k_name, (d_type, d_params) in inputs_data.items():
                        min_lim, max_lim = None, None
                        if 'poro' in k_name or 'fact_desv' in k_name: min_lim, max_lim = 0.0, 100.0
                        if k_name in ['perm', 'kh', 'kv', 'kx', 'kz', 'kf', 'h', 'rw', 're', 'visc', 'area', 'L', 'wf']: min_lim = 0.001
                        
                        user_min = d_params.pop('min_limit', None)
                        user_max = d_params.pop('max_limit', None)
                        
                        if user_min is not None: min_lim = user_min
                        if user_max is not None: max_lim = user_max
                        
                        v = generate_montecarlo(iterations, d_type, d_params, min_limit=min_lim, max_limit=max_lim)
                        if 'poro' in k_name or 'fact_desv' in k_name: v = v / 100.0 
                        vecs[k_name] = v

                    q_sim = np.zeros(iterations)
                    sys_arg = system # Ya viene como 'english' o 'international' desde dashboard.py
                    
                    if "Desviación Histórica" in model_type:
                        q_sim = ipr_aceite_desviacion_historica(vecs['q_det'], vecs['fact_desv'])
                    elif "2." in model_type:
                        q_sim = ipr_aceite_darcy(vecs['perm'], vecs['h'], vecs['dp'], vecs['bo'], vecs['visc'], vecs['re'], vecs['rw'], vecs['skin'], sys_arg)
                    elif "3." in model_type:
                        q_sim = ipr_aceite_darcy_empirico(vecs['j'], vecs['dp'])
                    elif "4." in model_type:
                        q_sim = ipr_aceite_darcy_ynf(vecs['perm'], vecs['kf'], vecs['poro_m'], vecs['poro_f'], vecs['h'], vecs['dp'], vecs['bo'], vecs['visc'], vecs['re'], vecs['rw'], vecs['skin'], sys_arg)
                    elif "5." in model_type:
                        q_sim = ipr_aceite_vogel(vecs['qtest'], vecs['dp_test'], vecs['pr_test'], vecs['dp'], vecs['pr'])
                    elif "6." in model_type:
                        q_sim = ipr_aceite_babu_odeh(vecs['kx'], vecs['kz'], vecs['L'], vecs['area'], vecs['dp'], vecs['bo'], vecs['visc'], vecs['rw'], sys_arg)
                    elif "7." in model_type:
                        q_sim = ipr_aceite_joshi(vecs['kh'], vecs['kv'], vecs['h'], vecs['L'], vecs['dp'], vecs['bo'], vecs['visc'], vecs['re'], vecs['rw'], sys_arg)
                    elif "8." in model_type:
                        q_sim = ipr_gas_pseudo_estable(vecs['perm'], vecs['h'], vecs['dp2'], vecs['visc'], vecs['z'], vecs['t'], vecs['re'], vecs['rw'], vecs['skin'], sys_arg)
                    elif "9." in model_type:
                        q_sim = ipr_gas_economides(vecs['perm'], vecs['kf'], vecs['wf'], vecs['xf'], vecs['h'], vecs['dp2'], vecs['visc'], vecs['z'], vecs['t'], vecs['re'], vecs['rw'], sys_arg)
                    elif "10." in model_type or "12." in model_type:
                        q_sim = ipr_gas_joshi_horizontal(vecs['kh'], vecs['kv'], vecs['h'], vecs['L'], vecs['dp2'], vecs['visc'], vecs['z'], vecs['t'], vecs['re'], vecs['rw'], vecs.get('skin', 0), sys_arg)
                    elif "11." in model_type:
                        q_sim = ipr_gas_pseudo_estable(vecs['perm'], vecs['h'], vecs['dp2'], vecs['visc'], vecs['z'], vecs['t'], vecs['re'], vecs['rw'], vecs['skin'], sys_arg)
                    elif "13." in model_type:
                        q_sim = ipr_gas_ynf(vecs['perm'], vecs['kf'], vecs['poro_m'], vecs['poro_f'], vecs['h'], vecs['dp2'], vecs['visc'], vecs['z'], vecs['t'], vecs['re'], vecs['rw'], vecs['skin'], sys_arg)
                    
                    q_sim = q_sim[np.isfinite(q_sim)]
                    
                    # Estadísticas
                    p90 = np.percentile(q_sim, 10) 
                    p50 = np.percentile(q_sim, 50)
                    p10 = np.percentile(q_sim, 90)
                    mean_q = np.mean(q_sim)
                    
                    # Caracterizar la distribución de q_sim
                    fit_df = fit_all_distributions(q_sim)
                    if not fit_df.empty:
                        best_row = fit_df.iloc[0]
                        dist_name = best_row['Distribution']
                        raw_params = best_row['_params_obj']
                        
                        mapped_params = {}
                        if dist_name == "Normal":
                            mapped_params = {'mu': raw_params[0], 'sigma': raw_params[1]}
                            clean_name = "Normal"
                        elif dist_name == "Lognormal (2P)":
                            mapped_params = {'mu': np.log(raw_params[2]), 'sigma': raw_params[0]}
                            clean_name = "Lognormal"
                        elif dist_name == "Weibull (2P)":
                            mapped_params = {'shape': raw_params[0], 'scale': raw_params[2]}
                            clean_name = "Weibull"
                        elif dist_name == "Gamma (2P)":
                            mapped_params = {'shape': raw_params[0], 'scale': raw_params[2]}
                            clean_name = "Gamma"
                        elif dist_name == "Exponential (1P)":
                            mapped_params = {'scale': raw_params[1]}
                            clean_name = "Exponencial"
                        elif dist_name == "Triangular":
                            c, loc, scale = raw_params
                            mapped_params = {'min': loc, 'mode': loc + c*scale, 'max': loc + scale}
                            clean_name = "Triangular"
                        elif dist_name == "Beta":
                            mapped_params = {'alpha': raw_params[0], 'beta': raw_params[1], 'min': raw_params[2], 'max': raw_params[2]+raw_params[3]}
                            clean_name = "Beta"
                        
                        st.session_state['qi_best_dist'] = clean_name
                        st.session_state['qi_best_params'] = mapped_params
                        
                        with st.expander("ℹ️ Caracterización Estocástica del Gasto Inicial", expanded=False):
                            st.markdown(f"**Mejor Ajuste (Anderson-Darling):** {dist_name}")
                            st.json(mapped_params)
                    
                    st.session_state['qi_sim'] = q_sim
                    
                    # Plotly Histogram
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=q_sim, 
                        histnorm='probability density',
                        marker_color='#cbd5e0',
                        opacity=0.75,
                        name='Frecuencia'
                    ))
                    
                    if not fit_df.empty:
                        best_dist_name = fit_df.iloc[0]['Distribution']
                        best_dist_params = fit_df.iloc[0]['_params_obj']
                        dist_obj = DISTRIBUTIONS[best_dist_name]
                        x_pdf = np.linspace(np.min(q_sim), np.max(q_sim), 200)
                        y_pdf = dist_obj.pdf(x_pdf, *best_dist_params)
                        fig.add_trace(go.Scatter(
                            x=x_pdf, y=y_pdf,
                            mode='lines',
                            line=dict(color='#2d3748', width=2),
                            name=f'Ajuste {best_dist_name}'
                        ))
                    
                    fig.add_vline(x=p90, line_dash="solid", line_color="#e53e3e", annotation_text=f"<b>P90</b><br>{int(p90)}", annotation_position="top left", annotation_font_color="#e53e3e")
                    fig.add_vline(x=p50, line_dash="solid", line_color="#3182ce", annotation_text=f"<b>P50</b><br>{int(p50)}", annotation_position="top left", annotation_font_color="#3182ce")
                    fig.add_vline(x=p10, line_dash="solid", line_color="#38a169", annotation_text=f"<b>P10</b><br>{int(p10)}", annotation_position="top left", annotation_font_color="#38a169")
                    
                    unidad_q = "STB/d" if fluid_type == "Oil" else "MCF/d"
                    
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        plot_bgcolor='white',
                        xaxis_title=f'Gasto Inicial ({unidad_q})',
                        yaxis_title='Densidad de Probabilidad',
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=500,
                        xaxis=dict(showgrid=True, gridcolor='#edf2f7'),
                        yaxis=dict(showgrid=True, gridcolor='#edf2f7', showticklabels=False)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tarjetas inferiores
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0;">
                            <div style="color: #718096; font-size: 11px; font-weight: bold; margin-bottom: 5px;">PROMEDIO (MEDIA)</div>
                            <div style="color: #2b6cb0; font-size: 24px; font-weight: bold;">{int(mean_q):,} <span style="font-size: 14px;">{unidad_q}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                        with c2:
                            dd_medio = np.mean(vecs.get('dp', np.array([1])))
                            if 'dp2' in vecs: dd_medio = np.sqrt(np.mean(vecs['dp2'])) # un proxy para mantener sentido de Indice J
                            j_index = mean_q / dd_medio if dd_medio > 0 else 0
                            st.markdown(f"""
                            <div style="background-color: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0;">
                                <div style="color: #718096; font-size: 11px; font-weight: bold; margin-bottom: 5px;">ÍNDICE DE PRODUCTIVIDAD</div>
                                <div style="color: #2b6cb0; font-size: 24px; font-weight: bold;">{j_index:.1f} <span style="font-size: 14px;">J</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0;">
                            <div style="color: #718096; font-size: 11px; font-weight: bold; margin-bottom: 5px;">RANGO DE CONFIANZA</div>
                            <div style="color: #2b6cb0; font-size: 24px; font-weight: bold;">{int(p10 - p90):,}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Sensibilidad (Spearman)
                    st.markdown("<hr style='margin: 30px 0px;'>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="font-weight: bold; color: #2d3748; font-size: 18px;">Sensibilidad (Spearman)</div>
                    <div style="color: #718096; font-size: 13px; margin-bottom: 20px;">Impacto de las variables de entrada en el Gasto Inicial</div>
                    """, unsafe_allow_html=True)
                    
                    corrs = {}
                    import scipy.stats as scipy_stats
                    for k_name, v_array in vecs.items():
                        if isinstance(v_array, np.ndarray) and len(v_array) > 1 and np.std(v_array) > 0:
                            corr, _ = scipy_stats.spearmanr(v_array, q_sim)
                            if not np.isnan(corr):
                                corrs[k_name] = corr
                    
                    if corrs:
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
                        
                    # Calculadora Probabilística
                    if not fit_df.empty:
                        st.markdown("<hr style='margin: 30px 0px;'>", unsafe_allow_html=True)
                        st.markdown("""
                        <div style="font-weight: bold; color: #2d3748; font-size: 18px;">Calculadora (Percentiles y Valores)</div>
                        <div style="color: #718096; font-size: 13px; margin-bottom: 20px;">Relación probabilística basada en la distribución ajustada</div>
                        """, unsafe_allow_html=True)
                        
                        calc_c1, calc_c2 = st.columns(2)
                        with calc_c1:
                            st.markdown("<div style='font-size: 14px; font-weight: 500; color: #4a5568;'>Percentil → Valor (PPF)</div>", unsafe_allow_html=True)
                            p_input = st.number_input("Percentil (%)", min_value=0.01, max_value=99.99, value=50.0, step=1.0, key="calc_p")
                            calc_val = dist_obj.ppf(p_input / 100.0, *best_dist_params)
                            st.success(f"Valor estimado a P{p_input}: **{calc_val:.2f} {unidad_q}**")
                            
                        with calc_c2:
                            st.markdown("<div style='font-size: 14px; font-weight: 500; color: #4a5568;'>Valor → Percentil (CDF)</div>", unsafe_allow_html=True)
                            v_input = st.number_input(f"Valor de la Variable ({unidad_q})", value=float(mean_q), step=10.0, key="calc_v")
                            calc_perc = dist_obj.cdf(v_input, *best_dist_params) * 100.0
                            st.info(f"Percentil estimado: **{calc_perc:.4f}%**")
                except Exception as e:
                    st.error(f"Error procesando la simulación: {str(e)}")
        else:
            st.info("Configure las distribuciones de las variables a la izquierda y presione 'Ejecutar Simulación IPR'.")
