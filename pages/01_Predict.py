import streamlit as st
import requests, os, json
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from src.schema import FEATURE_SCHEMA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

st.set_page_config(page_title='Predict', page_icon='🔮', layout='wide')
st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] { background-color: #0A0E1A !important; }
    h1, h2, h3, h4, h5, h6 { color: #E8ECF4 !important; }
    p, li, span, label { color: #E8ECF4; }
</style>
""", unsafe_allow_html=True)

st.title('🔮 Individual Prediction + Explanation')

with st.form('predict_form'):
    cols = st.columns(3)
    user = {}
    numeric = FEATURE_SCHEMA['numerical']
    categorical = FEATURE_SCHEMA['categorical']

    for i, (col, spec) in enumerate(numeric.items()):
        with cols[i % 3]:
            user[col] = st.number_input(col.replace('.', '_'), min_value=float(spec['min']), max_value=float(spec['max']), value=float((spec['min'] + spec['max']) / 2))

    for i, (col, options) in enumerate(categorical.items()):
        with cols[(i + len(numeric)) % 3]:
            user[col] = st.selectbox(col.replace('.', '_'), options)

    submitted = st.form_submit_button('Predict & Explain')

if submitted:
    payload = {k.replace('_', '.') if '.' in k else k: v for k, v in user.items()}
    # Fix aliases for API
    api_payload = {}
    for k, v in user.items():
        api_payload[k] = v
    try:
        r = requests.post(f'{API_URL}/api/v1/predict', json=api_payload, timeout=10)
        if r.status_code == 200:
            res = r.json()
            prob = res['probability']
            pred = res['prediction']
            st.success(f"**Prediction:** {'✅ Likely to Subscribe' if pred == 1 else '❌ Unlikely'} — Probability: **{prob:.1%}** (threshold {res['threshold']})")
        else:
            st.error(f'API error {r.status_code}: {r.text[:200]}')
    except Exception as e:
        st.error(f'API unreachable: {e}. Start it with `uvicorn src.api:app --reload`')

    # SHAP explanation (load model locally)
    try:
        pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))
        X_input = pd.DataFrame([user])
        X_transformed = pipeline.named_steps['preprocessor'].transform(X_input) if hasattr(pipeline, 'named_steps') else pipeline.estimator.named_steps['preprocessor'].transform(X_input)
        model = pipeline.named_steps['classifier'] if hasattr(pipeline, 'named_steps') else pipeline.estimator.named_steps['classifier']
        feature_names = (pipeline.named_steps['preprocessor'] if hasattr(pipeline, 'named_steps') else pipeline.estimator.named_steps['preprocessor']).get_feature_names_out()

        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_transformed)

        sv = explainer.shap_values(X_transformed)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(values=sv[0] if isinstance(sv, list) else sv[0], base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[0], data=X_transformed[0], feature_names=feature_names), max_display=10, show=False)
        st.pyplot(fig)
        plt.close()
        st.caption('↑ factors push probability UP, ↓ factors push it DOWN')
    except Exception as e:
        st.info(f'SHAP unavailable: {e}')
