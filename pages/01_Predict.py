import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl')

@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)

@st.cache_resource
def get_explainer(pipeline):
    # Cache the explainer so we don't recreate it every time!
    model = pipeline.named_steps['classifier']
    background = shap.sample(pipeline.named_steps['preprocessor'].transform(pd.DataFrame([X_train_sample])), 100)
    explainer = shap.Explainer(model, background)
    return explainer

pipeline = load_pipeline()

st.title('Prediction & Explainability')
st.write('Explaining predictions with cached SHAP values.')

# Simplified input form for demo
age = st.number_input('Age', 18, 100, 30)
job = st.selectbox('Job', ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'student', 'unknown'])
# ... add rest of inputs ...

if st.button('Predict'):
    user_df = pd.DataFrame([{'age': age, 'job': job}])
    X_transformed = pipeline.named_steps['preprocessor'].transform(user_df)
    
    prediction = pipeline.predict(user_df)[0]
    st.write(f'Prediction: {prediction}')
    
    # Use cached explainer
    explainer = get_explainer(pipeline)
    shap_values = explainer(X_transformed)
    
    # Plot with correct feature names
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=10)
    st.pyplot(plt)
