import streamlit as st

st.set_page_config(page_title="Bank Marketing ML App", layout="wide")

st.title("Bank Marketing Prediction System")
st.write("""
This app demonstrates an end-to-end ML pipeline: from raw data to deployment.
Select a page from the sidebar to explore.
""")

# Correct metrics based on the actual evaluation (Replacing old 89.4% accuracy)
col1, col2, col3 = st.columns(3)
col1.metric("Best F1 Score", "0.530")
col2.metric("PR-AUC", "0.491")
col3.metric("Precision@20%", "0.373")
