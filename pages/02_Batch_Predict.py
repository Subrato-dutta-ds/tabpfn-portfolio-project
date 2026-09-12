import streamlit as st
import pandas as pd
import requests
import os

API_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

st.set_page_config(page_title='Batch Predict', page_icon='📦', layout='wide')

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] { background-color: #0A0E1A !important; }
    h1, h2, h3, h4, h5, h6 { color: #E8ECF4 !important; }
    p, li, span, label { color: #E8ECF4; }
</style>
""", unsafe_allow_html=True)

st.title('📦 Batch Prediction')
st.caption('Upload a CSV of customers. The app will automatically map columns and call the API.')

# Column name mapping (dataset format -> API format)
COLUMN_MAP = {
    'emp.var.rate': 'emp_var_rate',
    'cons.price.idx': 'cons_price_idx',
    'cons.conf.idx': 'cons_conf_idx',
    'nr.employed': 'nr_employed',
}

REQUIRED_API_COLS = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
    'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome',
    'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed'
]

uploaded = st.file_uploader('Choose a CSV file', type='csv')

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(f'Loaded **{len(df):,}** rows with **{len(df.columns)}** columns')

    # Drop customer_id if present (API doesn't accept it)
    customer_ids = df['customer_id'].tolist() if 'customer_id' in df.columns else None
    df_for_api = df.drop(columns=['customer_id'], errors='ignore').copy()

    # Rename dataset columns to API-friendly names
    df_for_api = df_for_api.rename(columns=COLUMN_MAP)

    # Validate required columns
    missing = [c for c in REQUIRED_API_COLS if c not in df_for_api.columns]
    if missing:
        st.error(f'❌ Missing required columns: {missing}')
    elif len(df_for_api) > 5000:
        st.error(f'❌ Batch too large: {len(df_for_api):,} rows. Max is 5000.')
    else:
        st.success(f'✅ All {len(REQUIRED_API_COLS)} required columns present.')
        st.dataframe(df_for_api.head(), use_container_width=True)

        if st.button('🚀 Predict Batch'):
            with st.spinner(f'Sending {len(df_for_api):,} customers to API...'):
                records = df_for_api[REQUIRED_API_COLS].to_dict(orient='records')
                try:
                    response = requests.post(f'{API_URL}/api/v1/predict-batch', json={'data': records}, timeout=60)
                    if response.status_code == 200:
                        results = response.json()['results']
                        result_df = pd.DataFrame(results)
                        if customer_ids:
                            result_df.insert(0, 'customer_id', customer_ids)
                        st.success(f'✅ Predicted {len(result_df):,} customers')
                        st.dataframe(result_df.head(100), use_container_width=True)
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button('📥 Download Predictions (CSV)', csv, 'batch_predictions.csv', 'text/csv')
                    else:
                        st.error(f'❌ API Error {response.status_code}: {response.text[:300]}')
                except Exception as e:
                    st.error(f'❌ Connection failed: {e}')
                    st.info('Make sure the API is running: `uvicorn src.api:app --reload`')
