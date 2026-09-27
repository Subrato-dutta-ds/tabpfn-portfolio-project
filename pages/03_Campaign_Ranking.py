import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.business import rank_customers, optimal_k, optimal_k_uplift, campaign_summary

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="Campaign Optimizer", page_icon="💰", layout="wide")

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] { background-color: #0A0E1A !important; }
    h1, h2, h3, h4, h5, h6 { color: #E8ECF4 !important; }
    p, li, span, label, div { color: #E8ECF4; }
    .metric-card {
        background: linear-gradient(135deg, #161B2E 0%, #1E2540 100%);
        border: 1px solid #2A3352; border-left: 4px solid #00D9B8;
        border-radius: 12px; padding: 1.2rem; text-align: center;
    }
    .metric-label { color: #B8C1D9 !important; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #00D9B8 !important; font-size: 1.7rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("💰 Campaign Optimizer")
st.caption("Expected-Profit Optimal K — propensity or uplift ranking (K=0 allowed)")

# Load ACTIVE model (not candidate)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'production_model_active.pkl')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'production_model.pkl')
if not os.path.exists(MODEL_PATH):
    st.error("No production model. Run `python src/train_models.py`.")
    st.stop()
production = joblib.load(MODEL_PATH)

# Load uplift models if available
uplift_t_path = os.path.join(BASE_DIR, 'models', 'uplift_treatment.pkl')
uplift_c_path = os.path.join(BASE_DIR, 'models', 'uplift_control.pkl')
uplift_available = os.path.exists(uplift_t_path) and os.path.exists(uplift_c_path)

data_path = os.path.join(BASE_DIR, 'data', 'synthetic_campaign_customers.csv')
if not os.path.exists(data_path):
    st.warning("Run `python src/generate_campaign_data.py`.")
    st.stop()

customers = pd.read_csv(data_path)
customer_ids = customers['customer_id'].tolist()
features = customers.drop(columns=['customer_id'])

with st.spinner("Scoring propensity..."):
    p_prop = production.predict_proba(features)[:, 1]

uplift_scores = None
if uplift_available:
    uplift_t = joblib.load(uplift_t_path)
    uplift_c = joblib.load(uplift_c_path)
    with st.spinner("Computing uplift (T-learner)..."):
        p_t = uplift_t.predict_proba(features)[:, 1]
        p_c = uplift_c.predict_proba(features)[:, 1]
        uplift_scores = p_t - p_c

# Ranking method
options = ["Propensity (P(subscribe))"]
if uplift_available:
    options.append("Uplift (Incremental Effect) [SIMULATED]")
default_idx = 1 if uplift_available else 0
mode = st.sidebar.radio("🎯 Ranking Method", options, index=default_idx)
use_uplift = "Uplift" in mode and uplift_scores is not None
score_col = 'uplift' if use_uplift else 'probability'

if use_uplift:
    st.sidebar.warning("Uplift uses SYNTHETIC treatment — demonstration only.")

scored = pd.DataFrame({
    'customer_id': customer_ids,
    'probability': p_prop,
    'uplift': uplift_scores if uplift_scores is not None else 0.0,
})
scored = rank_customers(scored, score_col=score_col)
scored['rank'] = range(1, len(scored) + 1)

st.sidebar.header("💰 Economics (Rs)")
revenue = st.sidebar.number_input("Revenue per subscription", 100, 100000, 2000, 100)
cost = st.sidebar.number_input("Cost per contact", 1, 5000, 50, 5)
st.sidebar.markdown("---")
st.sidebar.caption("Manual K — 0 means contact nobody")
user_k = st.sidebar.slider("Manual K", 0, len(scored), min(100, len(scored)), 50)

if use_uplift:
    best_k, best_profit = optimal_k_uplift(scored['uplift'].values, revenue, cost)
    summary = campaign_summary(scored['uplift'].values, best_k, revenue, cost)
else:
    best_k, best_profit = optimal_k(scored['probability'].values, revenue, cost)
    summary = campaign_summary(scored['probability'].values, best_k, revenue, cost)

summary_user = campaign_summary(scored[score_col].values, user_k, revenue, cost)

label = "INCREMENTAL" if use_uplift else "expected"
st.markdown(f"### Ranking by **{score_col}** — Optimal K = **{best_k:,}**, {label} profit **Rs {best_profit:,.0f}**")

c1, c2, c3, c4 = st.columns(4)
vals = [f"{summary['contacts']:,}", f"{summary['expected_conversions']:.0f}",
        f"Rs {summary['expected_profit']:,.0f}", f"{summary['lift']:.2f}x"]
labels = ["Optimal K", f"{'Incr' if use_uplift else 'Exp'} Conversions",
          f"{'Incr' if use_uplift else 'Exp'} Profit",
          "Incremental Uplift@K" if use_uplift else "Lift"]
for col, l, v in zip([c1, c2, c3, c4], labels, vals):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{l}</div><div class="metric-value">{v}</div></div>', unsafe_allow_html=True)

st.markdown(f"> **Note:** K* = argmax over K=0..N of cumulative {score_col} × Rs {revenue:,} − K × Rs {cost:,}.")

st.markdown("---")
st.subheader(f"Your Selection: K = {user_k:,}")
c1, c2, c3, c4 = st.columns(4)
u_vals = [f"{summary_user['contacts']:,}", f"{summary_user['expected_conversions']:.0f}",
          f"Rs {summary_user['expected_profit']:,.0f}", f"{summary_user['lift']:.2f}x"]
for col, l, v in zip([c1, c2, c3, c4], ["Contacts", "Conversions", "Profit", "Incremental Uplift@K" if use_uplift else "Lift"], u_vals):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{l}</div><div class="metric-value">{v}</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("Profit Curve (K=0 to N)")
k_values = np.arange(0, len(scored) + 1)
cum = scored[score_col].cumsum().values
profits = np.concatenate([[0.0], cum * revenue - np.arange(1, len(scored) + 1) * cost])
step = max(1, len(k_values) // 200)
st.line_chart(pd.DataFrame({'K': k_values[::step], 'Profit': profits[::step]}).set_index('K'))

st.markdown("---")
st.subheader(f"Top 100 Ranked by {score_col}")
show_cols = ['rank', 'probability', 'uplift', 'customer_id']
show_cols = [c for c in show_cols if c in scored.columns]
merged = scored[show_cols].head(100).merge(customers[['customer_id', 'age', 'job']], on='customer_id', how='left')
st.dataframe(merged, use_container_width=True, hide_index=True)

csv = scored.merge(customers, on='customer_id', how='left').to_csv(index=False).encode('utf-8')
st.download_button('📥 Download Campaign List (CSV)', csv, 'campaign_contacts.csv', 'text/csv')
