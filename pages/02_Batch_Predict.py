import os
"""
Batch Prediction Page
"""
import streamlit as st
import pandas as pd
import requests
from utils import get_models, predict_batch

st.set_page_config(page_title="Batch Predict", page_icon="??", layout="wide")

st.title("?? Batch Prediction")
st.markdown("Upload a CSV file with feature values to get predictions in bulk.")

# Model selection
models, default_model = get_models()
if not models:
    st.warning("No models available from API.")
    selected_model = "XGBoost"
else:
    selected_model = st.selectbox("Choose model", models, index=models.index(default_model) if default_model in models else 0)

st.markdown("---")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="CSV must contain the same 19 features in order.")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if st.button("?? Run Batch Prediction", type="primary"):
            # Convert to list of lists
            features_list = df.values.tolist()
            with st.spinner("Predicting..."):
                result = predict_batch(features_list, selected_model)
                if result:
                    df_pred = df.copy()
                    df_pred["prediction"] = result["predictions"]
                    df_pred["probability"] = result["probabilities"]
                    st.success("? Batch prediction complete!")
                    st.dataframe(df_pred)
                    
                    # Download
                    csv = df_pred.to_csv(index=False).encode('utf-8')
                    st.download_button("?? Download Results", csv, "predictions.csv", "text/csv")
                else:
                    st.error("? Batch prediction failed. Check API logs.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a CSV file to begin.")

with st.expander("?? Sample CSV Format"):
    st.markdown("""
    Your CSV should have **19 numeric columns** in the exact order expected by the model.
    You can download a sample template:
    """)
    sample = pd.DataFrame([[30, 2, 1, 3, 0, 1, 0, 2, 5, 1, 1, 999, 0, 1, 0, 93, -40, 4.8, 5000]],
                          columns=["age","job","marital","education","default","housing","loan",
                                   "contact","month","day_of_week","campaign","pdays","previous",
                                   "poutcome","emp.var.rate","cons.price.idx","cons.conf.idx",
                                   "euribor3m","nr.employed"])
    st.dataframe(sample)
    csv = sample.to_csv(index=False).encode('utf-8')
    st.download_button("Download Sample CSV", csv, "sample_input.csv", "text/csv")
