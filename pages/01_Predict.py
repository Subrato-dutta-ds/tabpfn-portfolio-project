import streamlit as st
import requests
import os

API_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

st.title("Model Prediction")
with st.form("full_form"):
    age = st.number_input("Age", 18, 100, 30)
    job = st.selectbox("Job", ["admin.", "blue-collar", "technician", "services", "management", "retired", "student", "unknown"])
    marital = st.selectbox("Marital", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["basic.4y", "high.school", "university.degree", "professional.course", "unknown"])
    default = st.selectbox("Default", ["no", "yes", "unknown"])
    housing = st.selectbox("Housing", ["no", "yes", "unknown"])
    loan = st.selectbox("Loan", ["no", "yes", "unknown"])
    contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
    month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"])
    campaign = st.number_input("Campaign", 1, 50, 1)
    pdays = st.number_input("Pdays", 0, 999, 999)
    previous = st.number_input("Previous", 0, 50, 0)
    poutcome = st.selectbox("Poutcome", ["nonexistent", "failure", "success"])
    emp_var_rate = st.number_input("Emp. Var. Rate", -10.0, 10.0, 1.1)
    cons_price_idx = st.number_input("Cons. Price Idx", 0.0, 1000.0, 93.994)
    cons_conf_idx = st.number_input("Cons. Conf. Idx", -100.0, 100.0, -36.4)
    euribor3m = st.number_input("Euribor 3m", 0.0, 10.0, 4.857)
    nr_employed = st.number_input("Nr. Employed", 0.0, 6000.0, 5191.0)
    btn = st.form_submit_button("Predict")

if btn:
    data = {
        "age": int(age), "job": job, "marital": marital, "education": education, "default": default,
        "housing": housing, "loan": loan, "contact": contact, "month": month, "day_of_week": day_of_week,
        "campaign": int(campaign), "pdays": int(pdays), "previous": int(previous), "poutcome": poutcome,
        "emp_var_rate": float(emp_var_rate), "cons_price_idx": float(cons_price_idx), "cons_conf_idx": float(cons_conf_idx),
        "euribor3m": float(euribor3m), "nr_employed": float(nr_employed)
    }
    try:
        response = requests.post(f'{API_URL}/predict', json=data)
        if response.status_code == 200:
            res = response.json()
            st.success(f"Prediction: {res['prediction']} (Probability: {res['probability']:.2%})")
            st.info(f"Optimized Threshold used: {res['threshold']}")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Connection Error: {e}. Please run uvicorn src.api:app --reload first!")
