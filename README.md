# Bank Marketing Prediction System

## Problem & Business Objective
The bank wants to know which customers to contact to maximize subscription rates. With a highly imbalanced dataset (~11% positive), **Accuracy is misleading**; we prioritize **Recall and Precision@20%** to optimize marketing budget.

## Dataset
UCI Bank Marketing Dataset (semicolon separated, contains duration which is dropped to prevent data leakage).

## Preprocessing
Canonical ColumnTransformer used for Training, Validation, Testing, and Inference: StandardScaler for numericals, OneHotEncoder for categoricals.

## Models & Business Metrics
Evaluated Logistic Regression, Random Forest, XGBoost, and TabPFN. Primary metric: F1, Secondary: PR-AUC, Tertiary: Precision@20%.
*[Paste your actual results here after running evaluate.py]*

## Architecture
Streamlit UI -> FastAPI (Pydantic Validation) -> Saved ML Pipeline -> Prediction

## How to Run
1. pip install -r requirements.txt
2. python src/train_models.py
3. python src/evaluate.py
4. uvicorn src.api:app --reload
5. streamlit run streamlit_app.py

## Deployment
Docker Compose (FastAPI + Streamlit) and GitHub Actions CI/CD.
