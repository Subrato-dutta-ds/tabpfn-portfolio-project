import streamlit as st
import pandas as pd
import numpy as np
import os
from src.monitoring import compute_psi, load_prediction_logs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="Monitoring", page_icon="📡", layout="wide")

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] { background-color: #0A0E1A !important; }
    h1, h2, h3, h4, h5, h6 { color: #E8ECF4 !important; }
    p, li, span, label, div { color: #E8ECF4; }
    .metric-card { background: linear-gradient(135deg, #161B2E 0%, #1E2540 100%);
        border: 1px solid #2A3352; border-left: 4px solid #00D9B8;
        border-radius: 12px; padding: 1.2rem; text-align: center; }
    .metric-label { color: #B8C1D9 !important; font-size: 0.8rem; text-transform: uppercase; }
    .metric-value { color: #00D9B8 !important; font-size: 1.5rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("📡 Production Monitoring")

df = load_prediction_logs(os.path.join(BASE_DIR, 'logs', 'predictions.jsonl'))

if df.empty:
    st.warning("No predictions logged yet. Make predictions on the Predict page.")
    st.stop()

st.subheader("🎯 Prediction Drift")
c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip([c1, c2, c3, c4],
                            ["Total", "Mean Prob", "Positive Rate", "P90 Prob"],
                            [f"{len(df):,}", f"{df['probability'].mean():.3f}",
                             f"{df['prediction'].mean():.1%}",
                             f"{df['probability'].quantile(0.9):.3f}"]):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("🔍 Feature Drift (PSI)")
st.caption("PSI < 0.1 stable | 0.1-0.25 monitor | > 0.25 drift")

baseline_path = os.path.join(BASE_DIR, 'reports', 'X_test.csv')
if os.path.exists(baseline_path) and len(df) > 5:
    baseline = pd.read_csv(baseline_path)
    feats = pd.DataFrame(df['features'].tolist())
    num_feats = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
                 'cons.conf.idx', 'euribor3m', 'nr.employed']
    rows = []
    for f in num_feats:
        if f in baseline.columns and f in feats.columns:
            v = pd.to_numeric(feats[f], errors='coerce').dropna()
            if len(v) > 5:
                psi = compute_psi(baseline[f].values, v.values)
                rows.append({'Feature': f, 'PSI': round(psi, 4),
                             'Status': 'Stable' if psi < 0.1 else ('Monitor' if psi < 0.25 else 'DRIFT')})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Need ≥6 predictions and reports/X_test.csv")

st.markdown("---")
st.subheader("📈 Recent Predictions")
st.dataframe(df.tail(100)[['timestamp', 'probability', 'prediction']], use_container_width=True, hide_index=True)
