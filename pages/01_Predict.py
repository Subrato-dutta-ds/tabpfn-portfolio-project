import os
"""
Single Prediction Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import get_features, get_models, predict_single, get_shap_explanation, load_feature_names_fallback

st.set_page_config(page_title="Predict", page_icon="??", layout="wide")

st.title("?? Single Prediction")
st.markdown("Enter client features to predict subscription likelihood.")

# Get feature names
feature_names = get_features() or load_feature_names_fallback()
if not feature_names:
    st.error("? Could not load feature names. Make sure FastAPI is running.")
    st.stop()

# Model selection
models, default_model = get_models()
if not models:
    st.warning("No models available from API.")
    selected_model = "XGBoost"
else:
    selected_model = st.selectbox("Choose model", models, index=models.index(default_model) if default_model in models else 0)

st.markdown("---")

# Input features in two columns
col1, col2 = st.columns(2)
input_values = []
with col1:
    for i, name in enumerate(feature_names[:10]):
        val = st.number_input(f"{name}", value=0.0, step=0.1, key=f"single_{i}")
        input_values.append(val)
with col2:
    for i, name in enumerate(feature_names[10:]):
        idx = i + 10
        val = st.number_input(f"{name}", value=0.0, step=0.1, key=f"single_{idx}")
        input_values.append(val)

if st.button("?? Predict", type="primary"):
    if len(input_values) != len(feature_names):
        st.error(f"Please fill all {len(feature_names)} features.")
    else:
        with st.spinner("Predicting..."):
            result, elapsed = predict_single(input_values, selected_model)
            if result:
                st.success(f"? Prediction complete (took {elapsed:.3f}s)")
                pred = result["prediction"]
                proba = result["probability"]
                confidence = result["confidence"]
                model_used = result["model_used"]
                
                # Display results
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Prediction", "Subscribed ?" if pred == 1 else "Not Subscribed ?")
                with cols[1]:
                    st.metric("Probability", f"{proba:.2%}")
                with cols[2]:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                st.info(f"Model used: {model_used}")
                
                # SHAP explanation
                st.subheader("?? How features influenced this prediction")
                # For demonstration, we'll use a sample SHAP from index 0 (or we can compute online)
                # Since we have cached SHAP, we can show a waterfall for a similar profile.
                # Instead, we'll fetch a precomputed waterfall from reports.
                shap_img = "reports/shap_waterfall_sample.png"
                if os.path.exists(shap_img):
                    st.image(shap_img, caption="Example SHAP waterfall (for a different sample)")
                else:
                    st.info("Run `python src/explainability.py` to generate SHAP waterfall plots.")
            else:
                st.error("? Prediction failed. Check API logs.")

# Show a sample input
with st.expander("?? Feature descriptions"):
    st.markdown("""
    Features (19 total):
    - **age**: Age of client
    - **job**: Encoded job category
    - **marital**: Encoded marital status
    - **education**: Encoded education level
    - **default**: Has credit in default?
    - **housing**: Has housing loan?
    - **loan**: Has personal loan?
    - **contact**: Contact communication type
    - **month**: Last contact month (encoded)
    - **day_of_week**: Last contact day (encoded)
    - **campaign**: Number of contacts during campaign
    - **pdays**: Days since last contact from previous campaign
    - **previous**: Number of contacts before this campaign
    - **poutcome**: Outcome of previous campaign
    - **emp.var.rate**: Employment variation rate
    - **cons.price.idx**: Consumer price index
    - **cons.conf.idx**: Consumer confidence index
    - **euribor3m**: Euribor 3 month rate
    - **nr.employed**: Number of employees
    """)
