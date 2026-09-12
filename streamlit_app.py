import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Bank Marketing AI", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    /* --- GLOBAL DARK BACKGROUND --- */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0A0E1A !important;
    }

    /* --- SIDEBAR STYLING --- */
    [data-testid="stSidebar"] {
        background-color: #0D1220 !important;
        border-right: 1px solid #1E2540 !important;
    }
    [data-testid="stSidebar"] * {
        color: #E8ECF4 !important;
    }
    [data-testid="stSidebar"] a {
        color: #E8ECF4 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 0.75rem !important;
        border-radius: 8px !important;
        transition: all 0.2s !important;
    }
    [data-testid="stSidebar"] a:hover {
        background-color: #1E2540 !important;
        color: #00D9B8 !important;
    }
    [data-testid="stSidebar"] a[aria-current="page"] {
        background-color: #1E2540 !important;
        color: #00D9B8 !important;
        border-left: 3px solid #00D9B8 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #00D9B8 !important;
    }

    /* --- MAIN CONTENT --- */
    .main-header {
        background: linear-gradient(135deg, #00D9B8 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        color: #B8C1D9 !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #161B2E 0%, #1E2540 100%);
        border: 1px solid #2A3352;
        border-left: 4px solid #00D9B8;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .metric-label {
        color: #B8C1D9 !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #00D9B8 !important;
        font-size: 2rem;
        font-weight: 700;
    }
    .info-box {
        background: linear-gradient(135deg, #161B2E 0%, #1E2540 100%);
        border: 1px solid #2A3352;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .info-box h3 {
        color: #00D9B8 !important;
        margin-top: 0;
    }
    .info-box p, .info-box li, .info-box ul {
        color: #E8ECF4 !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    .info-box b {
        color: #00D9B8 !important;
    }
    .arch-box {
        background: #0D1117;
        border: 1px solid #2A3352;
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'Consolas', monospace;
        color: #00D9B8 !important;
        font-size: 0.95rem;
        line-height: 1.8;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #E8ECF4 !important;
    }
    p, li, span, label {
        color: #E8ECF4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 Bank Marketing AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">Predict → Rank → Optimize → Explain</div>', unsafe_allow_html=True)

try:
    comparison = pd.read_csv(os.path.join(BASE_DIR, 'reports', 'model_comparison.csv'))
    best = comparison.sort_values('PR-AUC', ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("BEST MODEL", best['Model']),
        ("PR-AUC", f"{best['PR-AUC']:.3f}"),
        ("F1 SCORE", f"{best['F1']:.3f}"),
        ("LIFT @ 20%", f"{best['Lift@20%']:.2f}x"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)
except Exception:
    st.warning("Run `python src/train_models.py` to generate metrics.")

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-box">
        <h3>🎯 What This Does</h3>
        <p>Given a limited marketing budget, this system answers:</p>
        <p><b>"Which customers should the bank contact?"</b></p>
        <ul>
            <li>💡 Predict individual subscription probability</li>
            <li>📦 Batch predict on a CSV of customers</li>
            <li>🏆 Rank customers by expected profit</li>
            <li>🔍 Explain predictions with SHAP</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
        <h3>🏗️ Architecture</h3>
        <div class="arch-box">
Streamlit UI<br>
&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
FastAPI (Pydantic)<br>
&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
model_pipeline.pkl<br>
&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
Probability → Threshold<br>
&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
Prediction
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 📈 Model Comparison")
try:
    comparison = pd.read_csv(os.path.join(BASE_DIR, 'reports', 'model_comparison.csv'))
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.caption("🏆 XGBoost selected based on highest PR-AUC (primary metric for imbalanced data).")
except Exception:
    st.info("Run `python src/train_models.py` to generate the comparison table.")

st.markdown("---")
st.caption("Built with Streamlit, FastAPI, XGBoost & SHAP | Data: UCI Bank Marketing (41,188 rows)")
