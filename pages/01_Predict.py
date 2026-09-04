import streamlit as st
import requests
import os

API_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

st.title("Predict")
with st.form('form'):
    age = st.number_input('Age', 18, 100, 30)
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'student', 'unknown'])
    # ... (add the remaining 17 features here) ...
    btn = st.form_submit_button('Predict')

if btn:
    data = {'age': age, 'job': job}
    try:
        response = requests.post(f'{API_URL}/predict', json=data)
        st.success(f"API Result: {response.json()}")
    except Exception as e:
        st.error(f"API Error: {e}. Please run uvicorn src.api:app --reload first!")
