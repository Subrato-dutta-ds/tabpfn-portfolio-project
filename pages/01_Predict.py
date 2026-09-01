import streamlit as st
import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt

# --- 1. DYNAMIC PATH SETUP (Reproducibility Fix) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl')
TABPFN_PATH = os.path.join(BASE_DIR, 'models', 'tabpfn_model.pkl')

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")
st.title("🔮 Model Prediction & Explainability")

# --- 2. LOAD MODELS (Cached for performance) ---
@st.cache_resource
def load_models():
    pipeline = None
    tabpfn = None
    if os.path.exists(PIPELINE_PATH):
        pipeline = joblib.load(PIPELINE_PATH)
    else:
        st.error("Pipeline not found! Run `python src/train_models.py` first.")
        
    if os.path.exists(TABPFN_PATH):
        tabpfn = joblib.load(TABPFN_PATH)
        
    return pipeline, tabpfn

pipeline, tabpfn_model = load_models()

# --- 3. USER INPUT FORM (Bank Marketing Dataset Features) ---
st.subheader("Enter Features for Prediction")

# Create the user input dictionary
# (Match these exactly to your raw dataset columns)
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ["admin.", "blue-collar", "technician", "services", "management", "retired", "student", "unknown"])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["basic.4y", "high.school", "university.degree", "professional.course", "unknown"])
    
    # Add the rest of your variables here (duration, campaign, etc.)
    # Make sure these match the column names in your dataset!
    
    # Collect into a dictionary
    user_input_dict = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        # ... include all other matching column names ...
    }
    
    # Model Selector (Integrating your provided logic)
    model_choice = st.selectbox("Select Model", ["Random Forest (Pipeline)", "TabPFN"])
    
    predict_button = st.form_submit_button("Predict")

# --- 4. PREDICTION AND REAL-TIME SHAP ---
if predict_button:
    if pipeline is None:
        st.stop()
        
    # Convert inputs to DataFrame
    user_df = pd.DataFrame([user_input_dict])
    
    # EXPLAIN RAW vs TRANSFORMED FEATURES
    st.write(f"**Raw Input Features:** {user_df.shape[1]}")
    
    # Transform the input using the pipeline
    X_transformed = pipeline.named_steps['preprocessor'].transform(user_df)
    
    st.write(f"**Transformed Features (One-Hot Encoded + Scaled):** {X_transformed.shape[1]}")
    st.markdown("---")
    
    # --- PREDICTION LOGIC (YOUR CODE) ---
    if model_choice == "TabPFN":
        if tabpfn_model is None:
            st.error("TabPFN model not found! Run `python src/evaluate.py` to train and save it.")
        else:
            st.subheader("Prediction Result (TabPFN)")
            # TabPFN requires transformed data!
            model = tabpfn_model
            prediction = model.predict(X_transformed)[0]
            probability = model.predict_proba(X_transformed)[0].tolist()
            
            if prediction == 1:
                st.success(f"Prediction: YES (Probability: {probability[1]:.2%})")
            else:
                st.error(f"Prediction: NO (Probability: {probability[0]:.2%})")
                
            st.warning("SHAP Waterfall is not supported for TabPFN in this version. Showing prediction only.")
            
    else: # Random Forest Pipeline
        st.subheader("Prediction Result (Random Forest)")
        model = pipeline
        prediction = model.predict(user_df)[0]
        probability = model.predict_proba(user_df)[0].tolist()
        
        if prediction == 1:
            st.success(f"Prediction: YES (Probability: {probability[1]:.2%})")
        else:
            st.error(f"Prediction: NO (Probability: {probability[0]:.2%})")
            
        st.markdown("---")
        
        # --- REAL-TIME SHAP (For RF Pipeline) ---
        st.subheader("Why was this prediction made?")
        try:
            # Explain THIS SPECIFIC input
            explainer = shap.Explainer(pipeline.named_steps['classifier'], X_transformed)
            shap_values = explainer(X_transformed)
            
            # Plot the waterfall
            plt.figure()
            shap.plots.waterfall(shap_values[0], max_display=10)
            plt.savefig(os.path.join(BASE_DIR, "reports", "current_prediction_shap.png"), bbox_inches='tight')
            plt.close()
            
            # Display the plot
            st.image(os.path.join(BASE_DIR, "reports", "current_prediction_shap.png"), 
                     caption="Feature contributions for THIS exact prediction", 
                     use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate SHAP plot: {e}")