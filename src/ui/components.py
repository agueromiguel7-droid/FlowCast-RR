import streamlit as st

def st_distribution_input(label, default_val, key_prefix):
    st.markdown(f"<div style='font-size: 13px; font-weight: 500; margin-bottom: 5px; color: #4a5568;'>Distribución {label}</div>", unsafe_allow_html=True)
    
    dist_type = st.selectbox(
        "Tipo", 
        ["Determinístico", "BetaPERT", "Lognormal", "Normal", "Triangular", "Weibull", "Gamma"], 
        index=1 if "Drawdown" in label or "Permeabilidad" in label else 0,
        key=f"{key_prefix}_dist",
        label_visibility="collapsed"
    )
    
    params = {}
    
    if dist_type == "Determinístico":
        val = st.number_input("Valor", value=float(default_val), key=f"{key_prefix}_val")
        params = {'value': val}
    elif dist_type == "BetaPERT":
        c1, c2, c3 = st.columns(3)
        min_v = c1.number_input("Mín", value=float(default_val*0.8), key=f"{key_prefix}_min")
        ml_v = c2.number_input("Moda", value=float(default_val), key=f"{key_prefix}_ml")
        max_v = c3.number_input("Máx", value=float(default_val*1.2), key=f"{key_prefix}_max")
        params = {'min': min_v, 'most_likely': ml_v, 'max': max_v}
    elif dist_type == "Lognormal":
        c1, c2 = st.columns(2)
        mu = c1.number_input("Media (ln)", value=0.0, key=f"{key_prefix}_mu")
        sigma = c2.number_input("Desv. Est. (ln)", value=1.0, key=f"{key_prefix}_sigma")
        params = {'mu': mu, 'sigma': sigma}
    elif dist_type == "Normal":
        c1, c2 = st.columns(2)
        mu = c1.number_input("Media", value=float(default_val), key=f"{key_prefix}_mu")
        sigma = c2.number_input("Desv. Est.", value=float(default_val*0.1), key=f"{key_prefix}_sigma")
        params = {'mu': mu, 'sigma': sigma}
    elif dist_type == "Triangular":
        c1, c2, c3 = st.columns(3)
        min_v = c1.number_input("Mín", value=float(default_val*0.8), key=f"{key_prefix}_min")
        ml_v = c2.number_input("Moda", value=float(default_val), key=f"{key_prefix}_ml")
        max_v = c3.number_input("Máx", value=float(default_val*1.2), key=f"{key_prefix}_max")
        params = {'min': min_v, 'most_likely': ml_v, 'max': max_v}
    elif dist_type == "Weibull":
        c1, c2 = st.columns(2)
        shape = c1.number_input("Forma (k)", value=1.5, key=f"{key_prefix}_shape")
        scale = c2.number_input("Escala (lambda)", value=float(default_val), key=f"{key_prefix}_scale")
        params = {'shape': shape, 'scale': scale}
    elif dist_type == "Gamma":
        c1, c2 = st.columns(2)
        shape = c1.number_input("Forma (k)", value=2.0, key=f"{key_prefix}_shape")
        scale = c2.number_input("Escala (theta)", value=float(default_val/2.0), key=f"{key_prefix}_scale")
        params = {'shape': shape, 'scale': scale}
        
    st.markdown("<hr style='margin: 10px 0px;'>", unsafe_allow_html=True)
    
    return dist_type.lower(), params
