import streamlit as st
import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl')
TABPFN_PATH = os.path.join(BASE_DIR, 'models', 'tabpfn_model.pkl')

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")
st.title("🔮 Model Prediction & Explainability")

@st.cache_resource
def load_models():
    pipeline = joblib.load(PIPELINE_PATH) if os.path.exists(PIPELINE_PATH) else None
    tabpfn = joblib.load(TABPFN_PATH) if os.path.exists(TABPFN_PATH) else None
    return pipeline, tabpfn

pipeline, tabpfn_model = load_models()

# --- FULL FEATURE FORM (ALL 19 FEATURES EXCEPT DURATION) ---
st.subheader("Enter Features for Prediction")
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ["admin.", "blue-collar", "technician", "services", "management", "retired", "student", "unknown"])
    marital = st.selectbox("Marital", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["basic.4y", "high.school", "university.degree", "professional.course", "unknown"])
    default = st.selectbox("Default", ["no", "yes", "unknown"])
    housing = st.selectbox("Housing", ["no", "yes", "unknown"])
    loan = st.selectbox("Loan", ["no", "yes", "unknown"])
    contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
    month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"])
    campaign = st.number_input("Campaign", min_value=1, value=1)
    pdays = st.number_input("Pdays", min_value=0, value=999)
    previous = st.number_input("Previous", min_value=0, value=0)
    poutcome = st.selectbox("Poutcome", ["nonexistent", "failure", "success"])
    emp_var_rate = st.number_input("Emp. Var. Rate", value=1.1)
    cons_price_idx = st.number_input("Cons. Price Idx", value=93.994)
    cons_conf_idx = st.number_input("Cons. Conf. Idx", value=-36.4)
    euribor3m = st.number_input("Euribor 3m", value=4.857)
    nr_employed = st.number_input("Nr. Employed", value=5191.0)
    
    user_input_dict = {
        "age": age, "job": job, "marital": marital, "education": education, "default": default,
        "housing": housing, "loan": loan, "contact": contact, "month": month, "day_of_week": day_of_week,
        "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
        "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx, "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m, "nr.employed": nr_employed
    }
    
    model_choice = st.selectbox("Select Model", ["Random Forest (Pipeline)", "TabPFN"])
    predict_button = st.form_submit_button("Predict")

if predict_button:
    if pipeline is None: st.stop()
    user_df = pd.DataFrame([user_input_dict])
    
    st.write(f"**Raw Input Features:** {user_df.shape[1]}")
    X_transformed = pipeline.named_steps['preprocessor'].transform(user_df)
    st.write(f"**Transformed Features:** {X_transformed.shape[1]}")
    
    if model_choice == "TabPFN":
        if tabpfn_model is None:
            st.error("TabPFN model not found! Run `python src/evaluate.py`.")
        else:
            prediction = tabpfn_model.predict(X_transformed)[0]
            probability = tabpfn_model.predict_proba(X_transformed)[0].tolist()
            st.success(f"TabPFN Prediction: YES ({probability[1]:.2%})" if prediction == 1 else f"TabPFN Prediction: NO ({probability[0]:.2%})")
            st.warning("SHAP not supported for TabPFN in this version.")
    else:
        prediction = pipeline.predict(user_df)[0]
        probability = pipeline.predict_proba(user_df)[0].tolist()
        st.success(f"RF Prediction: YES ({probability[1]:.2%})" if prediction == 1 else f"RF Prediction: NO ({probability[0]:.2%})")
        st.markdown("---")
        explainer = shap.Explainer(pipeline.named_steps['classifier'], X_transformed)
        shap_values = explainer(X_transformed)
        plt.figure()
        shap.plots.waterfall(shap_values[0], max_display=10)
        plt.savefig(os.path.join(BASE_DIR, "reports", "current_prediction_shap.png"), bbox_inches='tight')
        plt.close()
        st.image(os.path.join(BASE_DIR, "reports", "current_prediction_shap.png"), caption="Feature contributions for THIS exact prediction", use_container_width=True)