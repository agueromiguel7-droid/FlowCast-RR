import streamlit as st
import src.utils.styles as styles
from src.ui.login import render_login
from src.ui.dashboard import render_dashboard

st.set_page_config(
    page_title="Reliarisk FlowCast",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar estilos globales
styles.apply_global_styles()

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    render_login()
else:
    render_dashboard()
