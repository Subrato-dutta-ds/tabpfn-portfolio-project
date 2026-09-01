import streamlit as st
import pandas as pd
import os

# --- 1. DYNAMIC PATH SETUP (Reproducibility Fix) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERF_PATH = os.path.join(BASE_DIR, 'reports', 'performance_table.csv')
FEAT_IMP_PATH = os.path.join(BASE_DIR, 'reports', 'feature_importance.png')
SHAP_SUMMARY_PATH = os.path.join(BASE_DIR, 'reports', 'shap_summary.png')
WATERFALL_PATH = os.path.join(BASE_DIR, 'reports', 'shap_waterfall_sample.png')
TRAIN_PATH = os.path.join(BASE_DIR, 'reports', 'training_results.csv')

st.set_page_config(page_title="Reports", page_icon="📊", layout="wide")

st.title("📊 Model Reports & Visualizations")
st.markdown("View saved evaluation results and SHAP plots.")

# --- 2. Performance Table (Sorted by F1 Score) ---
st.subheader("🏆 Model Performance Comparison")
if os.path.exists(PERF_PATH):
    df = pd.read_csv(PERF_PATH)
    
    # Sort by F1 Score instead of Accuracy
    df = df.sort_values(by='F1 Score', ascending=False)
    st.dataframe(df, use_container_width=True)
    
    # Change your caption to prioritize F1
    st.caption("Best F1 Score: " + df.loc[df['F1 Score'].idxmax(), 'Model'])
    st.caption("**Note:** Given the imbalanced dataset, F1 and Recall are prioritized over Accuracy.")
else:
    st.info("Run `python src/evaluate.py` to generate performance table.")

st.markdown("---")

# --- 3. Feature Importance (SHAP) ---
st.subheader("🔍 Feature Importance (SHAP)")
col1, col2 = st.columns(2)
with col1:
    if os.path.exists(FEAT_IMP_PATH):
        st.image(FEAT_IMP_PATH, caption="Top 15 Features by Mean |SHAP|", use_container_width=True)
    else:
        st.info("Run `python src/explainability.py` to generate.")
with col2:
    if os.path.exists(SHAP_SUMMARY_PATH):
        st.image(SHAP_SUMMARY_PATH, caption="Global SHAP Summary", use_container_width=True)
    else:
        st.info("SHAP summary not generated yet.")

st.markdown("---")

# --- 4. Waterfall ---
st.subheader("💧 Individual Explanation (SHAP Waterfall)")
if os.path.exists(WATERFALL_PATH):
    st.image(WATERFALL_PATH, caption="SHAP Waterfall for a sample prediction", use_container_width=True)
else:
    st.info("Waterfall plot not available. Run `python src/explainability.py`.")

# --- 5. Training Results ---
st.subheader("📈 Training Results")
if os.path.exists(TRAIN_PATH):
    df_train = pd.read_csv(TRAIN_PATH)
    st.dataframe(df_train, use_container_width=True)
else:
    st.info("Training results not found. Run `python src/train_models.py`.")