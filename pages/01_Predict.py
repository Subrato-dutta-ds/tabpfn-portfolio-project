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
            default = float((spec['min'] + spec['max']) / 2)
            key = col.replace('.', '_')
            if col in ['age', 'campaign', 'pdays', 'previous']:
                user[col] = st.number_input(key, min_value=int(spec['min']), max_value=int(spec['max']), value=int(default))
            else:
                user[col] = st.number_input(key, min_value=float(spec['min']), max_value=float(spec['max']), value=float(default))

    for i, (col, options) in enumerate(categorical.items()):
        with cols[(i + len(numeric)) % 3]:
            user[col] = st.selectbox(col, options)

    submitted = st.form_submit_button('Predict & Explain')

if submitted:
    # Build API payload
    api_payload = {}
    for k, v in user.items():
        if k == 'emp.var.rate': api_payload['emp_var_rate'] = v
        elif k == 'cons.price.idx': api_payload['cons_price_idx'] = v
        elif k == 'cons.conf.idx': api_payload['cons_conf_idx'] = v
        elif k == 'nr.employed': api_payload['nr_employed'] = v
        else: api_payload[k] = v

    try:
        r = requests.post(f'{API_URL}/api/v1/predict', json=api_payload, timeout=10)
        if r.status_code == 200:
            res = r.json()
            st.success(f"**Prediction:** {'✅ Likely to Subscribe' if res['prediction'] == 1 else '❌ Unlikely'} — Probability: **{res['probability']:.1%}** (threshold {res['threshold']})")
        else:
            st.error(f'API error {r.status_code}: {r.text[:200]}')
    except Exception as e:
        st.error(f'API unreachable: {e}. Start it with `uvicorn src.api:app --reload`')

    # SHAP (load production model locally)
    try:
        production = joblib.load(os.path.join(BASE_DIR, 'models', 'production_model.pkl'))
        if hasattr(production, 'calibrated_classifiers_'):
            base_pipeline = production.calibrated_classifiers_[0].estimator
        else:
            base_pipeline = production
        preprocessor = base_pipeline.named_steps['preprocessor']
        model = base_pipeline.named_steps['classifier']
        feature_names = preprocessor.get_feature_names_out()

        X_input = pd.DataFrame([user])
        X_transformed = preprocessor.transform(X_input)

        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_transformed)

        sv = explainer.shap_values(X_transformed)
        if isinstance(sv, list): sv = sv[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[0],
                base_values=float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0]),
                data=X_transformed[0],
                feature_names=feature_names
            ),
            max_display=10, show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption('↑ factors push probability UP, ↓ factors push it DOWN')
        st.caption('SHAP — Base XGBoost Explanation. This explains the underlying base estimator. Displayed probability is the calibrated output, which may differ slightly.')
    except Exception as e:
        st.info(f'SHAP unavailable: {e}')
