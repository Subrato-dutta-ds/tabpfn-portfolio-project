import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Bank Marketing AI", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] { background-color: #0A0E1A !important; }
    [data-testid="stSidebar"] { background-color: #0D1220 !important; border-right: 1px solid #1E2540 !important; }
    [data-testid="stSidebar"] * { color: #E8ECF4 !important; }
    [data-testid="stSidebar"] a[aria-current="page"] { background-color: #1E2540 !important; color: #00D9B8 !important; }
    .main-header {
        background: linear-gradient(135deg, #00D9B8 0%, #3B82F6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 3rem; font-weight: 800; margin-bottom: 0.2rem;
    }
    .sub-caption { color: #B8C1D9 !important; font-size: 1.1rem; margin-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #161B2E 0%, #1E2540 100%);
        border: 1px solid #2A3352; border-left: 4px solid #00D9B8;
        border-radius: 12px; padding: 1.2rem; text-align: center;
    }
    .metric-label { color: #B8C1D9 !important; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #00D9B8 !important; font-size: 2rem; font-weight: 700; }
    .info-box {
        background: linear-gradient(135deg, #161B2E 0%, #1E2540 100%);
        border: 1px solid #2A3352; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;
    }
    .info-box h3 { color: #00D9B8 !important; margin-top: 0; }
    .info-box p, .info-box li { color: #E8ECF4 !important; line-height: 1.6; }
    h1, h2, h3, h4, h5, h6 { color: #E8ECF4 !important; }
    p, li, span, label { color: #E8ECF4; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 Bank Marketing AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">Predict → Rank → Optimize → Explain</div>', unsafe_allow_html=True)

# Read FINAL TEST results (single source of truth)
final_path = os.path.join(BASE_DIR, 'reports', 'final_test_results.csv')
if os.path.exists(final_path):
    final = pd.read_csv(final_path).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("MODEL", str(final['Model'])),
        ("TEST F1", f"{final['Test_F1']:.3f}"),
        ("TEST PR-AUC", f"{final['Test_PR_AUC']:.3f}"),
        ("LIFT @ 20%", f"{final['Lift@20%']:.2f}x"),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)
else:
    st.warning("Run `python src/train_models.py` then `python src/evaluate.py` to generate metrics.")

st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="info-box">
        <h3>🎯 What This Does</h3>
        <p>Given a limited marketing budget, this system answers:</p>
        <p><b style="color:#00D9B8">"Which customers should the bank contact?"</b></p>
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
        <pre style="color:#00D9B8; font-family: Consolas, monospace; font-size: 0.95rem; line-height: 1.6;">
Streamlit UI
    ↓
FastAPI (Pydantic)
    ↓
production_model.pkl
    ↓
Calibrated Probability
    ↓
Threshold → Prediction
    </pre>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📈 Validation Model Comparison")
comp_path = os.path.join(BASE_DIR, 'reports', 'model_comparison.csv')
if os.path.exists(comp_path):
    comp = pd.read_csv(comp_path)
    st.dataframe(comp, use_container_width=True, hide_index=True)
    best_model = comp.sort_values('Val_PR_AUC', ascending=False).iloc[0]['Model']
    st.caption(f"🏆 Selected: **{best_model}** (highest validation PR-AUC)")
else:
    st.info("Run `python src/train_models.py` to generate the comparison table.")

st.markdown("---")
st.caption("Built with Streamlit, FastAPI, XGBoost & SHAP | Data: UCI Bank Marketing (41,188 rows)")
