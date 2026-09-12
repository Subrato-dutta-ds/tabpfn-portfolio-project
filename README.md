# Bank Marketing Campaign Optimization

An end-to-end ML system for optimizing bank marketing campaigns: predicting subscription probability, ranking customers, and maximizing expected profit.

## Business Problem
Given a limited marketing budget, which customers should the bank contact?

## Data Leakage Handling
The duration feature was **removed** because it is only known after the marketing interaction and would not be available at campaign targeting time.

## Models Compared
- Logistic Regression
- Random Forest
- XGBoost
*(TabPFN was excluded due to a vendor license requirement that could not be fulfilled in this environment)*

## Results
Run python src/evaluate.py to generate the latest comparison. See eports/model_comparison.csv and eports/profit_analysis.csv.

**Business Interpretation:** Contacting the top 20% of customers by predicted probability captures approximately 3.3x the baseline subscription rate.

## Architecture
- **Streamlit UI** -> **FastAPI** -> **Saved ML Pipeline** -> **Prediction + Threshold**

## How to Run
1. pip install -r requirements.txt
2. python src/train_models.py
3. python src/evaluate.py
4. uvicorn src.api:app --reload
5. streamlit run streamlit_app.py

## Business Metrics
- **Precision@20%:** Of the top 20% contacted, how many actually subscribed
- **Lift@20%:** How much better than random targeting
- **Expected Profit:** sum(P_i * Revenue - Cost) for all selected customers
