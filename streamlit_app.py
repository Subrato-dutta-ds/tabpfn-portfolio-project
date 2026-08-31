"""
Main Streamlit app - Home page with navigation.
"""
import streamlit as st
from utils import check_api_health, get_models, get_features, load_feature_names_fallback
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1e3a5f;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #4a5568;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #3182ce;
        }
        .success-badge {
            background-color: #c6f6d5;
            color: #22543d;
            padding: 0.2rem 0.8rem;
            border-radius: 1rem;
            font-weight: 600;
        }
        .error-badge {
            background-color: #fed7d7;
            color: #9b2c2c;
            padding: 0.2rem 0.8rem;
            border-radius: 1rem;
            font-weight: 600;
        }
        .stButton button {
            width: 100%;
            background-color: #1e3a5f;
            color: white;
        }
        .stButton button:hover {
            background-color: #2a4a7f;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
st.sidebar.title("🏦 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔮 Single Predict", "📂 Batch Predict", "📊 Reports", "ℹ️ About"],
    index=0
)

# API health check
api_healthy, health_data = check_api_health()
if api_healthy:
    st.sidebar.success(f"✅ API Connected ({health_data.get('models_loaded', [])})")
else:
    st.sidebar.error("❌ API Not Connected")

st.sidebar.markdown("---")
st.sidebar.caption("Built with FastAPI + Streamlit")
st.sidebar.caption("© 2026 Portfolio Project")

# ---------- PAGE ROUTING ----------
if page == "🏠 Home":
    st.markdown('<p class="main-header">🏦 Bank Marketing Campaign</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict client subscription to term deposits using machine learning</p>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Available", len(get_models()[0]) if api_healthy else "N/A")
    with col2:
        st.metric("Features", len(get_features() or load_feature_names_fallback() or []))
    with col3:
        st.metric("Test Samples", "1,800")
    with col4:
        st.metric("Accuracy (Best)", "89.4%" if api_healthy else "N/A")
    
    st.markdown("---")
    
    # Quick overview
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📊 Dataset Overview")
        st.markdown("""
        - **Source**: UCI Bank Marketing dataset
        - **Instances**: 41,188 (subsampled to 9,000 for training)
        - **Features**: 19 (categorical encoded, scaled)
        - **Target**: `y` – subscribed to term deposit (yes/no)
        - **Class Balance**: ~11% positive (imbalanced)
        """)
    with col_right:
        st.subheader("🚀 Models Deployed")
        if api_healthy:
            models = get_models()[0]
            for m in models:
                st.markdown(f"- ✅ {m}")
        else:
            st.warning("Start FastAPI to see models")
    
    # Show a sample of SHAP summary if available
    st.subheader("📈 Feature Importance (SHAP)")
    shap_img = "reports/shap_summary.png"
    if os.path.exists(shap_img):
        st.image(shap_img, caption="Global SHAP summary plot")
    else:
        st.info("Run `python src/explainability.py` to generate SHAP plots")

elif page == "🔮 Single Predict":
    # This page will be handled by pages/01_Predict.py
    st.switch_page("pages/01_Predict.py")

elif page == "📂 Batch Predict":
    st.switch_page("pages/02_Batch_Predict.py")

elif page == "📊 Reports":
    st.switch_page("pages/03_Reports.py")

elif page == "ℹ️ About":
    st.switch_page("pages/04_About.py")
