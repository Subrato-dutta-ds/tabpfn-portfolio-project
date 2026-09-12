import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

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

st.title("💰 Budget-Based Campaign Optimizer")
st.caption("Enter a marketing budget → system contacts the top-K customers by calibrated probability")

pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'production_model.pkl'))
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
    proba = pipeline.predict_proba(features)[:, 1]

scored = pd.DataFrame({'customer_id': customer_ids, 'probability': proba})
scored = scored.sort_values('probability', ascending=False).reset_index(drop=True)
scored['rank'] = range(1, len(scored) + 1)

st.sidebar.header("💼 Budget & Economics (₹)")
revenue = st.sidebar.number_input("Revenue per subscription (₹)", 100, 100000, int(metadata.get('revenue_per_subscription', 2000)), 100)
cost = st.sidebar.number_input("Cost per contact (₹)", 1, 5000, int(metadata.get('cost_per_contact', 50)), 5)
budget = st.sidebar.number_input("Total campaign budget (₹)", 1000, 50000000, 100000, 1000)

max_contacts = int(budget // cost)
n_contact = min(max_contacts, len(scored))

st.markdown(f"### With **₹{budget:,}** budget → contact **{n_contact:,}** customers (at ₹{cost}/contact)")

top = scored.head(n_contact)
expected_conversions = top['probability'].sum()
expected_cost = n_contact * cost
expected_revenue = (top['probability'] * revenue).sum()
expected_profit = expected_revenue - expected_cost
roi = (expected_profit / expected_cost * 100) if expected_cost > 0 else 0
baseline_rate = proba.mean()
lift = (top['probability'].mean() / baseline_rate) if baseline_rate > 0 else 0

c1, c2, c3, c4 = st.columns(4)
for col, label, value in zip(
    [c1, c2, c3, c4],
    ["Contacts", "Expected Conversions", "Expected Profit", "ROI / Lift"],
    [f"{n_contact:,}", f"{expected_conversions:.0f}", f"₹{expected_profit:,.0f}", f"{roi:.0f}% / {lift:.2f}x"]
):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.markdown(f"""
> ⚠️ **Note:** Model-estimated expected values. Uses calibrated probabilities from isotonic regression.
> Assumes independence between customers. Revenue (₹{revenue:,}) and cost (₹{cost:,}) are user-defined.
> **Campaign selection = top-K by probability (no threshold).** Individual classification threshold is separate.
""")

st.markdown("---")
st.subheader("📈 Budget Sensitivity Analysis")
budgets = np.linspace(budget * 0.2, budget * 3, 15).astype(int)
rows = []
for b in budgets:
    k = min(int(b // cost), len(scored))
    if k == 0: continue
    top_k = scored.head(k)
    ep = float((top_k['probability'] * revenue).sum() - k * cost)
    rows.append({'Budget (₹)': int(b), 'Contacts': k, 'Expected Profit (₹)': round(ep, 0)})
sens_df = pd.DataFrame(rows)
st.dataframe(sens_df, use_container_width=True, hide_index=True)

best = sens_df.loc[sens_df['Expected Profit (₹)'].idxmax()]
st.success(f"💡 **Optimal budget: ₹{best['Budget (₹)']:,}** → contact {best['Contacts']:,} customers → expected profit ₹{best['Expected Profit (₹)']:,.0f}")

st.markdown("---")
st.subheader("🏆 Top Ranked Customers")
st.dataframe(scored.head(100).merge(customers[['customer_id', 'age', 'job']], on='customer_id', how='left'), use_container_width=True, hide_index=True)

csv = scored.merge(customers, on='customer_id', how='left').to_csv(index=False).encode('utf-8')
st.download_button('📥 Download Campaign List (CSV)', csv, 'campaign_contacts.csv', 'text/csv')
