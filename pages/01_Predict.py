import streamlit as st
import requests
import os

API_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

st.title('Predict')
st.write('This app calls the FastAPI backend.')

with st.form('form'):
    age = st.number_input('Age', 18, 100, 30)
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'student', 'unknown'])
    # Add the rest of your 19 features here later!
    btn = st.form_submit_button('Predict')

if btn:
    data = {'age': age, 'job': job}
    try:
        response = requests.post(f'{API_URL}/predict', json=data)
        st.success(f'API Response: {response.json()}')
    except Exception as e:
        st.error(f'API Error: {e}. Please run uvicorn src.api:app --reload first!')
