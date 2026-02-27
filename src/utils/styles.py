import streamlit as st

def apply_global_styles():
    st.markdown("""
    <style>
        /* General background */
        .stApp {
            background-color: #f4f6f8;
            font-family: 'Inter', 'Roboto', sans-serif;
        }
        
        /* Hide Default Streamlit UI elements for a cleaner desktop look */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Login Card Container */
        .login-card {
            background-color: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0px 10px 30px rgba(0,0,0,0.05);
            max-width: 420px;
            margin: auto;
            margin-top: 10vh;
            text-align: center;
        }
        
        /* Typography */
        .login-title {
            color: #1a202c;
            font-weight: 700;
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .login-subtitle {
            color: #718096;
            font-size: 14px;
            margin-bottom: 30px;
        }
        
        /* Submit Button custom style */
        .stButton>button {
            width: 100%;
            background-color: #2b6cb0;
            color: white;
            border-radius: 6px;
            padding: 8px 16px;
            border: none;
            font-weight: 600;
        }
        
        .stButton>button:hover {
            background-color: #2c5282;
            color: white;
            border-color: #2c5282;
        }
        
        /* Inputs */
        div[data-baseweb="input"] {
            border-radius: 6px;
        }
        
        /* Footer Text */
        .login-footer {
            text-align: center;
            margin-top: 50px;
            color: #a0aec0;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
    </style>
    """, unsafe_allow_html=True)
