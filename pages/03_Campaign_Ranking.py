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

st.title("💰 Campaign Optimizer — Expected-Profit Optimal K")
st.caption("Computes the expected-profit-optimal number of contacts by scanning all K from 1 to N.")

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
scored = scored.sort_values('probability', ascending=False).reset_index(drop=True)
scored['rank'] = range(1, len(scored) + 1)

st.sidebar.header("💼 Budget & Economics (₹)")
revenue = st.sidebar.number_input("Revenue per subscription (₹)", 100, 100000, int(metadata.get('revenue_per_subscription', 2000)), 100)
cost = st.sidebar.number_input("Cost per contact (₹)", 1, 5000, int(metadata.get('cost_per_contact', 50)), 5)

# Exact top-K optimal: scan all K from 1 to N
cumulative_proba = scored['probability'].cumsum().values
k_values = np.arange(1, len(scored) + 1)
profit_curve = cumulative_proba * revenue - k_values * cost

best_k = int(np.argmax(profit_curve)) + 1
best_profit = float(profit_curve[best_k - 1])

# Also show a user-selected K
st.sidebar.markdown("---")
user_k = st.sidebar.slider("Manual K (customers to contact)", 1, len(scored), best_k, 50)

# Manual metrics
top_user = scored.head(user_k)
profit_user = float((top_user['probability'].sum() * revenue) - user_k * cost)
conversions_user = float(top_user['probability'].sum())
baseline_rate = proba.mean()
lift_user = (top_user['probability'].mean() / baseline_rate) if baseline_rate > 0 else 0

# Optimal metrics
top_best = scored.head(best_k)
conversions_best = float(top_best['probability'].sum())
baseline_rate = proba.mean()
lift_best = (top_best['probability'].mean() / baseline_rate) if baseline_rate > 0 else 0

st.markdown(f"### 🎯 Expected-Profit Optimal: contact **{best_k:,}** customers for max profit **₹{best_profit:,.0f}**")

c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip(
    [c1, c2, c3, c4],
    ["Contacts (Optimal K)", "Expected Conversions", "Expected Profit", "Lift"],
    [f"{best_k:,}", f"{conversions_best:.0f}", f"₹{best_profit:,.0f}", f"{lift_best:.2f}x"]
):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

st.markdown(f"""
> ⚠️ **Note:** Exact optimum computed by scanning all K=1..{len(scored):,}.  
> Model-estimated expected values using calibrated probabilities.  
> Revenue (₹{revenue:,}) and cost (₹{cost:,}) are user assumptions.  
> The optimum is: `K* = argmax_K [ R * sum(top_K probs) - K * C ]`
""")

st.markdown("---")
st.subheader(f"📊 Your Selection: K = {user_k:,}")
c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip(
    [c1, c2, c3, c4],
    ["Contacts", "Expected Conversions", "Expected Profit", "Lift"],
    [f"{user_k:,}", f"{conversions_user:.0f}", f"₹{profit_user:,.0f}", f"{lift_user:.2f}x"]
):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("📈 Profit Curve (all K)")
plot_df = pd.DataFrame({
    'K': k_values[::max(1, len(k_values)//200)],
    'Expected_Profit': profit_curve[::max(1, len(k_values)//200)]
})
st.line_chart(plot_df.set_index('K'))
st.caption(f"Peak at K={best_k:,} (₹{best_profit:,.0f})")

st.markdown("---")
st.subheader("🏆 Top 100 Ranked Customers")
st.dataframe(scored.head(100).merge(customers[['customer_id', 'age', 'job']], on='customer_id', how='left'), use_container_width=True, hide_index=True)

csv = scored.merge(customers, on='customer_id', how='left').to_csv(index=False).encode('utf-8')
st.download_button('📥 Download Campaign List (CSV)', csv, 'campaign_contacts.csv', 'text/csv')
