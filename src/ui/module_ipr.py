import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.core.montecarlo import generate_montecarlo
from src.core.models_ipr import (
    ipr_aceite_desviacion_historica,
    ipr_aceite_darcy,
    ipr_aceite_darcy_empirico,
    ipr_aceite_vogel,
    ipr_aceite_babu_odeh,
    ipr_aceite_joshi,
    ipr_gas_pseudo_estable,
    ipr_gas_economides,
    ipr_gas_joshi_horizontal,
    ipr_gas_ynf
)
from src.ui.components import st_distribution_input

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
            "2. IPR Darcy - Método Analítico": {"Roca": ['perm', 'poro', 'h'], "Fluido": ['visc', 'bo'], "Presiones": ['dd', 're', 'rw', 'skin'], "Latex": r"q_o = \frac{0.00708 \cdot k \cdot h \cdot (P_r - P_{wf})}{B_o \cdot \mu_o \cdot \left(\ln\left(\frac{r_e}{r_w}\right) - 0.75 + s\right)}"},
            "3. IPR Darcy - Método Empírico": {"Roca": ['perm', 'h'], "Fluido": ['visc', 'bo'], "Presiones": ['dd'], "Latex": r"q_o = J \cdot (P_r - P_{wf}) \quad \text{donde} \quad J \approx \frac{0.00708 \cdot k \cdot h}{\mu_o \cdot B_o}"},
            "4. IPR Darcy Modificado (YNF) - Método Analítico": {"Roca": ['perm', 'kf', 'h'], "Fluido": ['visc', 'bo'], "Presiones": ['dd', 're', 'rw', 'skin'], "Latex": r"q_o = \frac{0.00708 \cdot k_f \cdot h \cdot (P_r - P_{wf})}{B_o \cdot \mu_o \cdot \left(\ln\left(\frac{r_e}{r_w}\right) - 0.75 + s\right)}"},
            "5. IPR-Vogel": {"Roca": ['perm', 'h'], "Fluido": ['visc', 'bo'], "Presiones": ['dd'], "Latex": r"\frac{q_o}{q_{max}} = 1 - 0.2\left(\frac{P_{wf}}{P_r}\right) - 0.8\left(\frac{P_{wf}}{P_r}\right)^2"},
            "6. IPR Babu&Odeh (Pozo Horizontal) - Método Analítico": {"Roca": ['perm', 'h'], "Fluido": ['visc', 'bo'], "Presiones": ['dd', 'len', 're', 'rw', 'skin'], "Latex": r"q_o = \frac{0.00708 \cdot L \cdot k \cdot (P_r - P_{wf})}{B_o \cdot \mu_o \cdot \left( \ln\left(\frac{\sqrt{A}}{r_w}\right) + \ln C_H - 0.75 + s_R \right)}"},
            "7. IPR Joshi (Pozo Horizontal) - Método Analítico": {"Roca": ['perm', 'h'], "Fluido": ['visc', 'bo'], "Presiones": ['dd', 'len', 're', 'rw'], "Latex": r"q_o = \frac{0.00708 \cdot k_h \cdot h \cdot (P_r - P_{wf})}{B_o \cdot \mu_o \cdot \left( \ln\left( \frac{a + \sqrt{a^2 - (L/2)^2}}{L/2} \right) + \frac{\beta^2 h}{L} \ln\left( \frac{h}{2r_w} \right) \right)}"},
            "8. Caudal en Estado Pseudo Estable": {"Roca": ['perm', 'poro', 'h'], "Fluido": ['visc', 'z', 't'], "Presiones": ['dd', 're', 'rw', 'skin'], "Latex": r"q_g = \frac{k \cdot h \cdot (P_r^2 - P_{wf}^2)}{1422 \cdot T \cdot \mu_g \cdot Z \cdot \left( \ln\left(\frac{r_e}{r_w}\right) - 0.75 + s \right)}"},
            "9. Gasto en Estado Pseudo Estable para Fracturamiento Hidráulico - Economides": {"Roca": ['perm', 'h'], "Fluido": ['visc', 'z', 't'], "Presiones": ['dd', 're', 'rw', 'skin'], "Latex": r"q_g = \frac{k \cdot h \cdot (P_r^2 - P_{wf}^2)}{1422 \cdot T \cdot \mu_g \cdot Z \cdot \left( \ln\left(\frac{r_e}{r_w}\right) - 0.75 + s + \Delta s_f \right)}"},
            "10. Gasto en Estado Estable - Pozo Horizontal - Joshi": {"Roca": ['perm', 'h'], "Fluido": ['visc', 'z', 't'], "Presiones": ['dd', 'len', 're', 'rw', 'skin'], "Latex": r"q_g = \frac{k_h \cdot h \cdot (P_r^2 - P_{wf}^2)}{1422 \cdot T \cdot \mu_g \cdot Z \cdot \left( \ln\left( \frac{a + \sqrt{a^2 - (L/2)^2}}{L/2} \right) + \frac{\beta^2 h}{L} \ln\left( \frac{h}{2r_w} \right) \right)}"},
            "11. Producción Commingled de arenas fracturadas": {"Roca": ['perm', 'h'], "Fluido": ['visc', 'z', 't'], "Presiones": ['dd', 're', 'rw', 'skin'], "Latex": r"q_{g,total} = \sum_{j=1}^{n} \frac{k_j \cdot h_j \cdot (P_r^2 - P_{wf}^2)}{1422 \cdot T \cdot \mu_g \cdot Z \cdot \left( \ln\left(\frac{r_e}{r_{w}}\right) - 0.75 + s_j \right)}"},
            "12. Gasto en Estado Estable - Pozo Horizontal con Fractura - Joshi": {"Roca": ['perm', 'h'], "Fluido": ['visc', 'z', 't'], "Presiones": ['dd', 'len', 're', 'rw', 'skin'], "Latex": r"q_g = \frac{k_h \cdot h \cdot (P_r^2 - P_{wf}^2)}{1422 \cdot T \cdot \mu_g \cdot Z \cdot f(L, x_f)}"},
            "13. Gasto en Estado Estable para Yacimientos Naturalmente Fracturados": {"Roca": ['perm', 'kf', 'h'], "Fluido": ['visc', 'z', 't'], "Presiones": ['dd', 're', 'rw', 'skin'], "Latex": r"q_g = \frac{k_f \cdot h \cdot (P_r^2 - P_{wf}^2)}{1422 \cdot T \cdot \mu_g \cdot Z \cdot \left( \ln\left(\frac{r_e}{r_w}\right) - 0.75 + s \right)}"}
        }
        
        reqs = model_reqs.get(model_type, {"Roca": ['perm', 'poro', 'h'], "Fluido": ['visc', 'bo', 'z', 't'], "Presiones": ['dd', 're', 'rw', 'skin'], "Latex": ""})
        
        if reqs["Latex"]:
            st.markdown("<div style='font-size: 13px; color: #718096; font-weight: bold;'>Modelo Matemático:</div>", unsafe_allow_html=True)
            st.latex(reqs["Latex"])
            st.markdown("<br>", unsafe_allow_html=True)
            
        inputs_data = {}
        
        if reqs["Roca"]:
            with st.expander("Propiedades de Roca", expanded=True):
                if 'perm' in reqs["Roca"]: inputs_data['perm'] = st_distribution_input("Permeabilidad (mD)", 150.0, "perm")
                if 'kf' in reqs["Roca"]: inputs_data['kf'] = st_distribution_input("Perm. Fractura (mD)", 500.0, "kf")
                if 'poro' in reqs["Roca"]: inputs_data['poro'] = st_distribution_input("Porosidad (%)", 18.0, "poro")
                if 'h' in reqs["Roca"]: inputs_data['h'] = st_distribution_input("Espesor Neto (ft)", 50.0, "esp")
            
        if reqs["Fluido"]:
            with st.expander("Propiedades de Fluido", expanded=True):
                if 'visc' in reqs["Fluido"]: inputs_data['visc'] = st_distribution_input("Viscosidad (cp)", 1.2, "visc")
                if fluid_type == "Oil" and 'bo' in reqs["Fluido"]:
                    inputs_data['bo'] = st_distribution_input("Factor Volumétrico Bo", 1.05, "bo")
                if fluid_type == "Gas":
                    if 'z' in reqs["Fluido"]: inputs_data['Z'] = st_distribution_input("Factor Z (Gas)", 0.85, "z")
                    if 't' in reqs["Fluido"]: inputs_data['T'] = st_distribution_input("Temperatura (°R)", 600.0, "t")
            
        if reqs["Presiones"]:
            with st.expander("Presiones y Otros", expanded=True):
                if 'dd' in reqs["Presiones"]: inputs_data['drawdown'] = st_distribution_input("Drawdown (Pr - Pwf) [psi]", 1000.0, "dd")
                if 're' in reqs["Presiones"]: inputs_data['re']   = st_distribution_input("Radio de Drenaje (ft)", 1500.0, "re")
                if 'rw' in reqs["Presiones"]: inputs_data['rw']   = st_distribution_input("Radio de Pozo (ft)", 0.328, "rw")
                if 'skin' in reqs["Presiones"]: inputs_data['skin'] = st_distribution_input("Daño (Skin)", 0.0, "skin")
                if 'len' in reqs["Presiones"]: inputs_data['L'] = st_distribution_input("Longitud Horizontal (ft)", 3000.0, "len")
                if 'q_det' in reqs["Presiones"]: inputs_data['q_det'] = st_distribution_input("Gasto Puntual Estimado", 1000.0, "q_det")
                if 'fact_desv' in reqs["Presiones"]: inputs_data['fact_desv'] = st_distribution_input("Factor Desviación", 0.1, "fdesv")
                
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
                        # Truncamiento físico: porosidad entre 0 y 100
                        min_lim, max_lim = None, None
                        if k_name == 'poro': min_lim, max_lim = 0.0, 100.0
                        if k_name in ['perm', 'h', 'rw', 're', 'visc', 'kf']: min_lim = 0.001
                        
                        v = generate_montecarlo(iterations, d_type, d_params, min_limit=min_lim, max_limit=max_lim)
                        vecs[k_name] = v

                    # Identificar modelo y calcular Q
                    q_sim = np.zeros(iterations)
                    sys_arg = system
                    
                    if "Desviación Histórica" in model_type:
                        q_sim = ipr_aceite_desviacion_historica(vecs['q_det'], vecs['fact_desv'])
                    elif "Darcy - Método Analítico" in model_type:
                        # Asumiendo Pr y Pwf no requeridos directo ya que drawdown=Pr-Pwf. 
                        # ipr_aceite_darcy needs Pr, Pwf to make Pr-Pwf. We pass 0 for pwf, and drawdown for pr.
                        q_sim = ipr_aceite_darcy(vecs['perm'], vecs['h'], vecs['drawdown'], 0, vecs['bo'], vecs['visc'], vecs['re'], vecs['rw'], vecs['skin'], sys_arg)
                    elif "Darcy - Método Empírico" in model_type:
                        # Empírico we need J: Let's assume J is entered via perm text somehow, or we use a standard proxy
                        J_proxy = (0.00708 * vecs['perm'] * vecs['h']) / (vecs['visc'] * vecs['bo'])  # Simplification
                        q_sim = ipr_aceite_darcy_empirico(J_proxy, vecs['drawdown'], 0)
                    elif "Darcy Modificado" in model_type: # YNF
                        q_sim = ipr_gas_ynf(vecs['perm'], vecs.get('kf', vecs['perm']*5), vecs['h'], vecs['drawdown'], 0, vecs['visc'], 1.0, 1.0, vecs['re'], vecs['rw'], vecs['skin'], 0.5)
                    elif "Vogel" in model_type:
                        qmax_proxy = (0.00708 * vecs['perm'] * vecs['h'] * vecs['drawdown']) / (vecs['visc'] * vecs['bo'] * 0.8) # proxy Qmax
                        # Needs Pr and Pwf separately for Vogel ratio
                        pr_assumed = vecs['drawdown'] + 1000 
                        pwf_assumed = 1000
                        q_sim = ipr_aceite_vogel(qmax_proxy, pr_assumed, pwf_assumed)
                    elif "Babu&Odeh" in model_type:
                        A_proxy = np.pi * vecs['re']**2
                        q_sim = ipr_aceite_babu_odeh(vecs['perm'], vecs['perm']*0.1, A_proxy, vecs['drawdown'], 0, vecs['bo'], vecs['visc'], vecs['rw'], vecs['skin'], 1.0)
                    elif "Joshi" in model_type and fluid_type == "Oil":
                        q_sim = ipr_aceite_joshi(vecs['perm'], vecs['h'], vecs['L'], vecs['drawdown'], 0, vecs['bo'], vecs['visc'], vecs['re'], vecs['rw'])

                    elif "Pseudo Estable" in model_type and "Economides" not in model_type:
                        # Gas
                        # Pr^2 - Pwf^2 = (Pr-Pwf)(Pr+Pwf) = Drawdown * (Drawdown + 2*Pwf). For simplicity Pr=drawdown, Pwf=0 (Pr^2 - Pwf^2) = drawdown^2 roughly or assume standard Pr.
                        pr_2_pwf_2 = vecs['drawdown'] * (vecs['drawdown'] + 2*1000) # approximation if single drawdown used
                        q_sim = ipr_gas_pseudo_estable(vecs['perm'], vecs['h'], np.sqrt(pr_2_pwf_2), 0, vecs['visc'], vecs['Z'], vecs['T'], vecs['re'], vecs['rw'], vecs['skin'])
                    elif "Economides" in model_type:
                        pr_2_pwf_2 = vecs['drawdown'] * (vecs['drawdown'] + 2*1000)
                        q_sim = ipr_gas_economides(vecs['perm'], vecs['h'], np.sqrt(pr_2_pwf_2), 0, vecs['visc'], vecs['Z'], vecs['T'], vecs['re'], vecs['rw'], vecs['skin'])
                    elif "Caudal en Estado Estable" in model_type and "Joshi" in model_type:
                        pr_2_pwf_2 = vecs['drawdown'] * (vecs['drawdown'] + 2*1000)
                        q_sim = ipr_gas_joshi_horizontal(vecs['perm'], vecs['h'], np.sqrt(pr_2_pwf_2), 0, vecs['visc'], vecs['Z'], vecs['T'], vecs['re'], vecs.get('L', 3000), vecs['h']*0.5, vecs['re'], vecs['rw'], vecs['skin'])
                    else:
                        # Default fallback
                        st.warning("El modelo seleccionado estricto será agregado en futuras iteraciones. Fallback a Darcy/PseudoEstable básico.")
                        q_sim = ipr_aceite_darcy(vecs.get('perm', np.array([10])), vecs.get('h', np.array([10])), vecs.get('drawdown', np.array([1000])), 0, vecs.get('bo', np.array([1.0])), vecs.get('visc', np.array([1.0])), vecs.get('re', np.array([500])), vecs.get('rw', np.array([0.5])), vecs.get('skin', np.array([0.0])), sys_arg)
                    
                    # Filtrar validos
                    q_sim = q_sim[np.isfinite(q_sim)]
                    
                    # Estadísticas
                    p90 = np.percentile(q_sim, 10) 
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
                    
                    fig.add_vline(x=p90, line_dash="solid", line_color="#e53e3e", annotation_text=f"<b>P90</b><br>{int(p90)}", annotation_position="top left", annotation_font_color="#e53e3e")
                    fig.add_vline(x=p50, line_dash="solid", line_color="#3182ce", annotation_text=f"<b>P50</b><br>{int(p50)}", annotation_position="top left", annotation_font_color="#3182ce")
                    fig.add_vline(x=p10, line_dash="solid", line_color="#38a169", annotation_text=f"<b>P10</b><br>{int(p10)}", annotation_position="top left", annotation_font_color="#38a169")
                    
                    unidad_q = "STB/d" if fluid_type == "Oil" else "MCF/d"
                    
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        plot_bgcolor='white',
                        xaxis_title=f'Gasto Inicial ({unidad_q})',
                        yaxis_title='Frecuencia Probabilística',
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
                        <div style="background-color: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0;">
                            <div style="color: #718096; font-size: 11px; font-weight: bold; margin-bottom: 5px;">PROMEDIO (MEDIA)</div>
                            <div style="color: #2b6cb0; font-size: 24px; font-weight: bold;">{int(mean_q):,} <span style="font-size: 14px;">{unidad_q}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        # Drawdown medio
                        dd_medio = np.mean(vecs.get('drawdown', [0]))
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
                except Exception as e:
                    st.error(f"Error procesando la simulación: {str(e)}")
        else:
            st.info("Configure las distribuciones de las variables a la izquierda y presione 'Ejecutar Simulación IPR'.")
