import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERF_PATH = os.path.join(BASE_DIR, 'reports', 'performance_table.csv')

st.title("Model Reports")
if os.path.exists(PERF_PATH):
    df = pd.read_csv(PERF_PATH)
    # Fix: Column is 'F1', not 'F1 Score'
    df = df.sort_values(by='F1', ascending=False)
    st.dataframe(df, use_container_width=True)
    st.caption("Best F1: " + df.loc[df['F1'].idxmax(), 'Model'])
    st.caption("Note: Given the imbalanced dataset, F1 and Precision@20% are prioritized.")
else:
    st.info("Run python src/evaluate.py to generate performance table.")
