import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from src.business import rank_customers, optimal_k, campaign_summary

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

st.title("💰 Campaign Optimizer — Expected-Profit Optimal K")
st.caption("Ranks customers by calibrated probability, then finds K* that maximizes expected profit.")

production = joblib.load(os.path.join(BASE_DIR, 'models', 'production_model.pkl'))
with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json')) as f:
    metadata = json.load(f)

data_path = os.path.join(BASE_DIR, 'data', 'synthetic_campaign_customers.csv')
if not os.path.exists(data_path):
    st.warning("Run `python src/generate_campaign_data.py` first.")
    st.stop()

customers = pd.read_csv(data_path)
customer_ids = customers['customer_id'].tolist()
features = customers.drop(columns=['customer_id'])

with st.spinner("Scoring customers..."):
    proba = production.predict_proba(features)[:, 1]

scored = pd.DataFrame({'customer_id': customer_ids, 'probability': proba})
scored = rank_customers(scored)
scored['rank'] = range(1, len(scored) + 1)

st.sidebar.header("Budget & Economics (Rs)")
revenue = st.sidebar.number_input("Revenue per subscription (Rs)", 100, 100000, int(metadata.get('revenue_per_subscription', 2000)), 100)
cost = st.sidebar.number_input("Cost per contact (Rs)", 1, 5000, int(metadata.get('cost_per_contact', 50)), 5)
st.sidebar.markdown("---")
user_k = st.sidebar.slider("Manual K (contacts)", 1, len(scored), 100, 50)

best_k, best_profit = optimal_k(scored['probability'].values, revenue, cost)
summary_best = campaign_summary(scored['probability'].values, best_k, revenue, cost)
summary_user = campaign_summary(scored['probability'].values, user_k, revenue, cost)

st.markdown(f"### Expected-Profit Optimal: contact **{best_k:,}** customers for max profit **Rs {best_profit:,.0f}**")

c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip(
    [c1, c2, c3, c4],
    ["Optimal K", "Expected Conversions", "Expected Profit", "Lift"],
    [f"{summary_best['contacts']:,}", f"{summary_best['expected_conversions']:.0f}",
     f"Rs {summary_best['expected_profit']:,.0f}", f"{summary_best['lift']:.2f}x"]
):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

st.markdown(f"""
> **Note:** Optimal K computed by scanning K=1..{len(scored):,}: `K* = argmax_K [ R * sum(top_K p_i) - K * C ]`.  
> Uses calibrated probabilities. Revenue (Rs {revenue:,}) and cost (Rs {cost:,}) are user-defined assumptions.  
> **This is expected-profit-optimal, not universally optimal.**
""")

st.markdown("---")
st.subheader(f"Your Selection: K = {user_k:,}")
c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip(
    [c1, c2, c3, c4],
    ["Contacts", "Expected Conversions", "Expected Profit", "Lift"],
    [f"{summary_user['contacts']:,}", f"{summary_user['expected_conversions']:.0f}",
     f"Rs {summary_user['expected_profit']:,.0f}", f"{summary_user['lift']:.2f}x"]
):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("Profit Curve (all K)")
k_values = np.arange(1, len(scored) + 1)
cum = scored['probability'].cumsum().values
profits = cum * revenue - k_values * cost
step = max(1, len(k_values) // 200)
st.line_chart(pd.DataFrame({'K': k_values[::step], 'Profit': profits[::step]}).set_index('K'))

st.markdown("---")
st.subheader("Top 100 Ranked Customers")
st.dataframe(scored.head(100).merge(customers[['customer_id', 'age', 'job']], on='customer_id', how='left'),
             use_container_width=True, hide_index=True)
csv = scored.merge(customers, on='customer_id', how='left').to_csv(index=False).encode('utf-8')
st.download_button('Download Campaign List (CSV)', csv, 'campaign_contacts.csv', 'text/csv')
