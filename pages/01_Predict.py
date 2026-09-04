import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))

st.title('Prediction & Explainability')

with st.form('form'):
    age = st.number_input('Age', 18, 100, 30)
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'student', 'unknown'])
    marital = st.selectbox('Marital', ['married', 'single', 'divorced'])
    education = st.selectbox('Education', ['basic.4y', 'high.school', 'university.degree', 'professional.course', 'unknown'])
    default = st.selectbox('Default', ['no', 'yes', 'unknown'])
    housing = st.selectbox('Housing', ['no', 'yes', 'unknown'])
    loan = st.selectbox('Loan', ['no', 'yes', 'unknown'])
    contact = st.selectbox('Contact', ['cellular', 'telephone', 'unknown'])
    month = st.selectbox('Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.selectbox('Day of Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
    campaign = st.number_input('Campaign', 1, 100, 1)
    pdays = st.number_input('Pdays', 0, 999, 999)
    previous = st.number_input('Previous', 0, 100, 0)
    poutcome = st.selectbox('Poutcome', ['nonexistent', 'failure', 'success'])
    emp_var_rate = st.number_input('Emp. Var. Rate', -10.0, 10.0, 1.1)
    cons_price_idx = st.number_input('Cons. Price Idx', 0.0, 1000.0, 93.994)
    cons_conf_idx = st.number_input('Cons. Conf. Idx', -100.0, 100.0, -36.4)
    euribor3m = st.number_input('Euribor 3m', 0.0, 10.0, 4.857)
    nr_employed = st.number_input('Nr. Employed', 0.0, 6000.0, 5191.0)
    btn = st.form_submit_button('Predict')

if btn:
    user_df = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
        'housing': housing, 'loan': loan, 'contact': contact, 'month': month, 'day_of_week': day_of_week,
        'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m, 'nr.employed': nr_employed
    }])
    
    prediction = pipeline.predict(user_df)[0]
    probability = pipeline.predict_proba(user_df)[0].tolist()
    st.write(f'Prediction: {prediction} (Prob: {probability[1]:.2%})')

    # Correct SHAP implementation (Fixes crash)
    X_transformed = pipeline.named_steps['preprocessor'].transform(user_df)
    explainer = shap.Explainer(pipeline.named_steps['classifier'], X_transformed)
    shap_values = explainer(X_transformed)
    
    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=10)
    plt.savefig(os.path.join(BASE_DIR, 'reports', 'temp_shap.png'), bbox_inches='tight')
    plt.close()
    st.image(os.path.join(BASE_DIR, 'reports', 'temp_shap.png'), caption='Feature contributions', use_container_width=True)
