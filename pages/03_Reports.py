"""
Reports Page - Show saved visualizations
"""
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Reports", page_icon="??", layout="wide")

st.title("?? Model Reports & Visualizations")
st.markdown("View saved evaluation results and SHAP plots.")

# Performance table
st.subheader("?? Model Performance Comparison")
perf_path = "reports/performance_table.csv"
if os.path.exists(perf_path):
    df = pd.read_csv(perf_path)
    st.dataframe(df, use_container_width=True)
    # Highlight best
    st.caption("Best accuracy: " + df.loc[df['Accuracy'].idxmax(), 'Model'])
else:
    st.info("Run `python src/evaluate.py` to generate performance table.")

st.markdown("---")

# Feature importance
st.subheader("?? Feature Importance (SHAP)")
col1, col2 = st.columns(2)
with col1:
    img_path = "reports/feature_importance.png"
    if os.path.exists(img_path):
        # FIXED: use_container_width=True
        st.image(img_path, caption="Top 15 Features by Mean |SHAP|", use_container_width=True)
    else:
        st.info("Run `python src/explainability.py` to generate.")
with col2:
    img_path2 = "reports/shap_summary.png"
    if os.path.exists(img_path2):
        # FIXED: use_container_width=True
        st.image(img_path2, caption="Global SHAP Summary", use_container_width=True)
    else:
        st.info("SHAP summary not generated yet.")

st.markdown("---")

# Waterfall
st.subheader("?? Individual Explanation (SHAP Waterfall)")
waterfall = "reports/shap_waterfall_sample.png"
if os.path.exists(waterfall):
    # FIXED: use_container_width=True
    st.image(waterfall, caption="SHAP Waterfall for a sample prediction", use_container_width=True)
else:
    st.info("Waterfall plot not available. Run `python src/explainability.py`.")

# Training results
st.subheader("?? Training Results")
train_path = "reports/training_results.csv"
if os.path.exists(train_path):
    df_train = pd.read_csv(train_path)
    st.dataframe(df_train, use_container_width=True)
else:
    st.info("Training results not found. Run `python src/train_models.py`.")