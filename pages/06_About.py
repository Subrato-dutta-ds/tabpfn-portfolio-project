import os
"""
About Page
"""
import streamlit as st

st.set_page_config(page_title="About", page_icon="??", layout="wide")

st.title("?? About This Project")
st.markdown("---")

st.markdown("""
## ?? Project Overview

This is a portfolio project demonstrating end-to-end machine learning engineering:

- **Business Problem**: Predict whether a client will subscribe to a term deposit during a bank marketing campaign.
- **Dataset**: UCI Bank Marketing dataset (~41k records, 20 features).
- **Models**: Logistic Regression, Random Forest, XGBoost (and optionally TabPFN).
- **Key Skills Demonstrated**:
  - Data preprocessing and leakage prevention
  - Model training and hyperparameter tuning
  - Rigorous evaluation (accuracy, F1, ROC-AUC, latency)
  - Explainability using SHAP
  - API development with FastAPI
  - Interactive dashboard with Streamlit
  - Containerization with Docker

## ??? Tech Stack

| Component | Technology |
|-----------|------------|
| Data Processing | Pandas, NumPy, Scikit-learn |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Deployment | AWS / GCP (optional) |

## ?? Project Structure
tabpfn-portfolio-project/
+-- api/ # FastAPI server
¦ +-- main.py
¦ +-- schemas.py
+-- src/ # ML pipeline
¦ +-- data_loader.py
¦ +-- train_models.py
¦ +-- evaluate.py
¦ +-- explainability.py
+-- models/ # Saved models & scaler
+-- reports/ # Plots & CSVs
+-- pages/ # Streamlit pages
+-- streamlit_app.py # Main app
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
## ?? Key Findings

- **Best Accuracy**: Logistic Regression (89.4%)
- **Best F1**: XGBoost (0.34) – handles class imbalance better
- **Fastest Inference**: Logistic Regression (0.0008 sec per 1k rows) – 15x faster than XGBoost
- **Key Features**: `euribor3m`, `emp.var.rate`, and `cons.conf.idx` have highest influence.

## ????? Author

Portfolio project built for data science / ML engineering interviews.

## ?? References

- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
""")
