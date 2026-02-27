import pandas as pd
import streamlit as st

# URL exportada a CSV de Google Sheets para usuarios y contraseñas
# El usuario debe reemplazar este valor con el CSV publicado de Drive.
GOOGLE_SHEET_CSV_URL = "YOUR_GOOGLE_SHEETS_CSV_PULIC_URL"

@st.cache_data(ttl=300)
def fetch_users():
    """
    Descarga el listado de usuarios desde un Google Sheet exportado como CSV.
    Si ocurre un error o la URL es un placeholder, usa credenciales de prueba.
    """
    try:
        if GOOGLE_SHEET_CSV_URL == "YOUR_GOOGLE_SHEETS_CSV_PULIC_URL":
            # Credenciales por defecto si no se configura URL
            return pd.DataFrame([{"usuario": "admin", "password": "123"}])
        
        df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
        # Limpiar espacios en blanco
        df['usuario'] = df['usuario'].astype(str).str.strip()
        df['password'] = df['password'].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error al conectar con base de datos de usuarios: {str(e)}")
        return None

def authenticate(username, password):
    """
    Valida las credenciales contra el DataFrame.
    """
    df = fetch_users()
    if df is not None and not df.empty:
        # Match usuario y contraseña (case-sensitive o dependiente del diseño)
        user_row = df[(df['usuario'] == username) & (df['password'] == password)]
        if not user_row.empty:
            return True
    return False
