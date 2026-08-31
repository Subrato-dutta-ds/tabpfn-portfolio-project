# TabPFN Portfolio Project

## Problem Statement
*Add a brief description of the problem (e.g., Predicting if a bank client will subscribe to a term deposit).*

## Dataset
*Add a brief description of the dataset (Bank Marketing dataset from UCI).*

## Approach
1. **Preprocessing:** Uses a saved ColumnTransformer pipeline (OneHotEncoder + StandardScaler) to ensure exact consistency between training and inference.
2. **Modeling:** Compares Logistic Regression, Random Forest, and XGBoost, with optional TabPFN.
3. **Explainability:** Generates real-time SHAP waterfall plots for individual predictions.

## Results
*Insert your final accuracy/F1 scores here (e.g., 91.08% accuracy).*

## Architecture
Streamlit UI -> FastAPI -> Saved ML Pipeline (joblib)

## Demo
*(Add a screenshot or GIF of your UI here)*

## How to Run
1. pip install -r requirements.txt
2. python src/train_models.py (Trains and saves the pipeline)
3. streamlit run streamlit_app.py

## Tech Stack
Python, Streamlit, FastAPI, Scikit-learn, SHAP, TabPFN, Docker.
