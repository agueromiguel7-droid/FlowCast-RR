import streamlit as st
import base64
from pathlib import Path
from src.utils.auth import authenticate

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def render_login():
    """
    Renderiza la interfaz de inicio de sesión similar al diseño de la imagen solicitada.
    """
    logo_path = Path("app_logo.png")
    
    # Check if logo exists to render it as base64 HTML
    if logo_path.exists():
        logo_b64 = image_to_base64(logo_path)
        img_html = f'<img src="data:image/png;base64,{logo_b64}" height="60" style="margin-bottom:20px;"/>'
    else:
        img_html = "<h2>🌊 FlowCast Logo</h2>"

    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        # Login card container
        st.markdown(f"""
        <div class="login-card">
            {img_html}
            <div class="login-title">Reliarisk FlowCast</div>
            <div class="login-subtitle">Reliability Analytics Suite</div>
        </div>
        """, unsafe_allow_html=True)
        
        # We put the form slightly overlapping the bottom of the card or inside 
        # By using standard Streamlit components, we place them right under the HTML block logically
        # To make it look like part of the card, we wrap the inputs in a form
        with st.form("login_form", clear_on_submit=False):
            st.markdown("**Usuario**")
            username = st.text_input("Usuario", placeholder="Ingrese su usuario", label_visibility="collapsed")
            
            st.markdown("**Contraseña**")
            password = st.text_input("Contraseña", type="password", placeholder="Ingrese su contraseña", label_visibility="collapsed")
            
            # Recordarme layout and forgot password
            c1, c2 = st.columns([1, 1])
            with c1:
                st.checkbox("Recordarme")
            with c2:
                st.markdown('<div style="text-align: right; color: #2b6cb0; font-size: 13px; font-weight: 600; padding-top: 5px;">¿Olvidó su contraseña?</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("Iniciar Sesión ➔")
            
            if submit_button:
                if authenticate(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos. (Prueba admin / 123)")
                    
        st.markdown('<div class="login-footer">© 2026 RELIARISK INC. SECURE LOGIN SYSTEM</div>', unsafe_allow_html=True)
