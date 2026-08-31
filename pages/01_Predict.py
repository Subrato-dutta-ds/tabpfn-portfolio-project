import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")

st.title("🔮 Model Prediction & Explainability")

# Load the trained pipeline
pipeline_path = "models/model_pipeline.pkl"
if os.path.exists(pipeline_path):
    pipeline = joblib.load(pipeline_path)
else:
    st.error("Model not found! Please run `python src/train_models.py` first.")
    st.stop()

# --- USER INPUT FORM ---
st.subheader("Enter Features for Prediction")

# Create the user input dictionary here
# (Replace these with your actual Streamlit widgets - selectbox, number_input, etc.)
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ["admin.", "blue-collar", "technician", "services", "management", "retired", "student", "unknown"])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["basic.4y", "high.school", "university.degree", "professional.course", "unknown"])
    # ... add the rest of your features here ...
    
    # Collect into a dictionary
    user_input_dict = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        # ... include all other matching column names ...
    }
    
    predict_button = st.form_submit_button("Predict")

# --- PREDICTION AND REAL-TIME SHAP ---
if predict_button:
    # 1. Convert inputs to DataFrame
    user_df = pd.DataFrame([user_input_dict])
    
    # 2. Make the prediction
    prediction = pipeline.predict(user_df)[0]
    probability = pipeline.predict_proba(user_df)[0].tolist()
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Prediction: YES (Probability: {probability[1]:.2%})")
    else:
        st.error(f"Prediction: NO (Probability: {probability[0]:.2%})")
    
    st.markdown("---")
    
    # 3. Explain THIS SPECIFIC prediction using SHAP
    st.subheader("Why was this prediction made?")
    try:
        # Transform the exact user input using the saved pipeline
        X_transformed = pipeline.named_steps['preprocessor'].transform(user_df)
        
        # Create SHAP explainer on the trained model
        explainer = shap.Explainer(pipeline.named_steps['classifier'], X_transformed)
        shap_values = explainer(X_transformed)
        
        # Plot the waterfall
        plt.figure()
        shap.plots.waterfall(shap_values[0], max_display=10)
        plt.savefig("reports/current_prediction_shap.png", bbox_inches='tight')
        plt.close()
        
        # Display the plot using the FIXED parameter
        st.image("reports/current_prediction_shap.png", 
                 caption="Feature contributions for THIS exact prediction", 
                 use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not generate SHAP plot: {e}")