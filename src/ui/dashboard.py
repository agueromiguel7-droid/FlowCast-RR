import streamlit as st
import base64
from pathlib import Path
from src.ui.module_ipr import render_ipr_module
from src.ui.module_dca import render_dca_module
from src.ui.module_char import render_characterization_module # New Support Module

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def render_dashboard():
    # Sidebar
    with st.sidebar:
        # Logo and Title
        logo_path = Path("mi_logo.png")
        if logo_path.exists():
            img_b64 = get_base64_of_bin_file(logo_path)
            st.markdown(f'''
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{img_b64}" width="30" style="margin-right: 10px;">
                <div style="font-weight: bold; font-size: 18px; color: #1a202c;">Reliarisk</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown("### Reliarisk FlowCast")
            
        st.markdown("<div style='font-size: 11px; font-weight: bold; color: #a0aec0; margin-bottom: 10px;'>INFORMACIÓN DEL CASO</div>", unsafe_allow_html=True)
        
        if 'regiones' not in st.session_state:
            st.session_state['regiones'] = ["Latam", "Norteamérica", "Europa"]
        if 'activos' not in st.session_state:
            st.session_state['activos'] = ["Flow-01", "Vaca Muerta - Bloque A", "Permian"]
            
        with st.expander("📍 Ubicación", expanded=True):
            region_sel = st.selectbox("Región", st.session_state['regiones'] + ["-- Agregar Nueva --"])
            if region_sel == "-- Agregar Nueva --":
                region_new = st.text_input("Nueva Región")
                if region_new and st.button("Guardar Región"):
                    st.session_state['regiones'].append(region_new)
                    st.rerun()
                region = region_new
            else:
                region = region_sel
                
            activo_sel = st.selectbox("Activo", st.session_state['activos'] + ["-- Agregar Nuevo --"])
            if activo_sel == "-- Agregar Nuevo --":
                activo_new = st.text_input("Nuevo Activo")
                if activo_new and st.button("Guardar Activo"):
                    st.session_state['activos'].append(activo_new)
                    st.rerun()
                activo = activo_new
            else:
                activo = activo_sel
            
        with st.expander("📊 Detalles del Análisis"):
            fecha = st.date_input("Fecha")
            analista = st.text_input("Analista", value=st.session_state.get('username', ''))
            
        st.markdown("<br><div style='font-size: 11px; font-weight: bold; color: #a0aec0; margin-bottom: 10px;'>CONFIGURACIÓN GLOBAL</div>", unsafe_allow_html=True)
        sistema_unidades = st.radio("Sistema de Unidades", ["Internacional", "Inglés"], index=1, horizontal=True)
        
        st.markdown("<br><div style='font-size: 11px; font-weight: bold; color: #a0aec0; margin-bottom: 10px;'>VARIABLES GLOBALES</div>", unsafe_allow_html=True)
        fluido = st.selectbox("FLUID TYPE", ["Oil", "Gas"])
        
        if fluido == "Oil":
            modelos_disponibles = [
                "1. Desviación Histórica",
                "2. IPR Darcy - Método Analítico",
                "3. IPR Darcy - Método Empírico",
                "4. IPR Darcy Modificado (YNF) - Método Analítico",
                "5. IPR-Vogel",
                "6. IPR Babu&Odeh (Pozo Horizontal) - Método Analítico",
                "7. IPR Joshi (Pozo Horizontal) - Método Analítico"
            ]
        else:
            modelos_disponibles = [
                "8. Caudal en Estado Pseudo Estable",
                "9. Gasto en Estado Pseudo Estable para Fracturamiento Hidráulico - Economides",
                "10. Gasto en Estado Estable - Pozo Horizontal - Joshi",
                "11. Producción Commingled de arenas fracturadas",
                "12. Gasto en Estado Estable - Pozo Horizontal con Fractura - Joshi",
                "13. Gasto en Estado Estable para Yacimientos Naturalmente Fracturados"
            ]
            
        modelo = st.selectbox("MODEL", modelos_disponibles)
        
        st.markdown("<div style='font-size: 11px; font-weight: bold; color: #a0aec0; margin-bottom: 10px; margin-top: 15px;'>SIMULACIÓN (ITERACIONES)</div>", unsafe_allow_html=True)
        iteraciones_str = st.radio("Iteraciones", ["1k", "5k", "10k"], horizontal=True, label_visibility="collapsed")
        iteraciones_map = {"1k": 1000, "5k": 5000, "10k": 10000}
        iteraciones = iteraciones_map[iteraciones_str]
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Cerrar Sesión", use_container_width=True):
            st.session_state['authenticated'] = False
            st.rerun()

    # Main content
    # Top header bar
    # Use columns to separate title and user info
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown("### Reliarisk FlowCast - Dashboard de Análisis <span style='background-color: #c6f6d5; color: #22543d; font-size: 10px; padding: 2px 6px; border-radius: 4px; vertical-align: middle;'>LIVE DATA</span>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div style='text-align: right; font-size: 14px;'><b>{st.session_state.get('username', 'Ingeniero')}</b><br><span style='color: gray; font-size: 11px;'>Lead Analyst</span></div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 5px;'>", unsafe_allow_html=True)
    
    # Custom Tabs
    tab0, tab1, tab2 = st.tabs(["[Soporte] Módulo 0: Caracterización", "Módulo I: Afluencia (IPR)", "Módulo II: Pronóstico (DCA)"])
    
    sys_str = "english" if sistema_unidades == "Inglés" else "international"
    
    with tab0:
        render_characterization_module()
        
    with tab1:
        render_ipr_module(fluido, modelo, iteraciones, sys_str)
        
    with tab2:
        render_dca_module(fluido, iteraciones)
