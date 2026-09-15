import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="Reports", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] { background-color: #0A0E1A !important; }
    h1, h2, h3, h4, h5, h6 { color: #E8ECF4 !important; }
    p, li, span, label, div { color: #E8ECF4; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Model Reports")

# Final test results (single source of truth for production model)
final_path = os.path.join(BASE_DIR, 'reports', 'final_test_results.csv')
if os.path.exists(final_path):
    st.subheader("🏁 Final Test Results (Production Model)")
    final = pd.read_csv(final_path)
    st.dataframe(final, use_container_width=True, hide_index=True)
    st.caption("Evaluated once on the untouched 20% test set using the frozen calibrated production model.")
else:
    st.info("Run `python src/evaluate.py` to generate final_test_results.csv")

st.markdown("---")

# Model comparison (validation-based selection)
comp_path = os.path.join(BASE_DIR, 'reports', 'model_comparison.csv')
if os.path.exists(comp_path):
    st.subheader("📈 Validation Model Comparison")
    comp = pd.read_csv(comp_path).sort_values('Val_PR_AUC', ascending=False)
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.caption(f"🏆 Selected: **{comp.iloc[0]['Model']}** (highest Val PR-AUC)")
else:
    st.info("Run `python src/train_models.py` to generate model_comparison.csv")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 Calibration Curve")
    calib_path = os.path.join(BASE_DIR, 'reports', 'calibration_curve.png')
    if os.path.exists(calib_path):
        st.image(calib_path, use_container_width=True)
    else:
        st.info("Run `python src/evaluate.py` to generate calibration curve.")

with col2:
    st.subheader("🔍 SHAP Summary")
    shap_path = os.path.join(BASE_DIR, 'reports', 'shap_summary.png')
    if os.path.exists(shap_path):
        st.image(shap_path, use_container_width=True)
    else:
        st.info("Run `python src/explainability.py` to generate SHAP summary.")

st.markdown("---")
st.subheader("⭐ Top Features (SHAP)")
fi_path = os.path.join(BASE_DIR, 'reports', 'feature_importance.csv')
if os.path.exists(fi_path):
    fi = pd.read_csv(fi_path).head(15)
    st.dataframe(fi, use_container_width=True, hide_index=True)
else:
    st.info("Run `python src/explainability.py` to generate feature importance.")
