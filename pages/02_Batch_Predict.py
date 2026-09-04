import streamlit as st
import pandas as pd
import requests
import os

API_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
EXPECTED_COLUMNS = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

st.title('Batch Predict')
st.write('Upload a CSV file for batch predictions.')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Validate Columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
    else:
        # Validate Types (basic check for numeric columns)
        numeric_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for NaN
        if df.isnull().sum().sum() > 0:
            st.warning("Data contains missing values. Please clean your data.")
        else:
            st.dataframe(df.head())
            if st.button('Predict Batch'):
                # Convert to list of dicts
                data = df.to_dict(orient='records')
                try:
                    response = requests.post(f'{API_URL}/predict-batch', json={'data': data})
                    if response.status_code == 200:
                        results = response.json()['results']
                        st.success('Prediction complete!')
                        st.write(pd.DataFrame(results))
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
