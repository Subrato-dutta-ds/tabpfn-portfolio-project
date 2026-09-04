import os, joblib, pandas as pd, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))
with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json')) as f:
    metadata = json.load(f)
best_threshold = metadata['threshold']

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Bank Prediction API", "docs": "/docs"}

@app.get("/models")
def get_models():
    return {"model": metadata['model'], "threshold": best_threshold}

@app.get("/features")
def get_features():
    return list(pipeline.named_steps['preprocessor'].get_feature_names_out())

@app.get("/health")
def health_check():
    return {"status": "ok"}

class CustomerFeatures(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

def map_to_dataset(data: CustomerFeatures):
    return {"age": data.age, "job": data.job, "marital": data.marital, "education": data.education, "default": data.default, "housing": data.housing, "loan": data.loan, "contact": data.contact, "month": data.month, "day_of_week": data.day_of_week, "campaign": data.campaign, "pdays": data.pdays, "previous": data.previous, "poutcome": data.poutcome, "emp.var.rate": data.emp_var_rate, "cons.price.idx": data.cons_price_idx, "cons.conf.idx": data.cons_conf_idx, "euribor3m": data.euribor3m, "nr.employed": data.nr_employed}

@app.post('/predict')
def predict(data: CustomerFeatures):
    try:
        df = pd.DataFrame([map_to_dataset(data)])
        probability = pipeline.predict_proba(df)[0, 1]
        # Fix #2: Apply optimized threshold
        prediction = int(probability >= best_threshold)
        return {'prediction': prediction, 'probability': probability, 'threshold': best_threshold}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class BatchFeatures(BaseModel):
    data: List[CustomerFeatures]

@app.post('/predict-batch')
def predict_batch(batch: BatchFeatures):
    try:
        df = pd.DataFrame([map_to_dataset(item) for item in batch.data])
        probabilities = pipeline.predict_proba(df)[:, 1]
        predictions = (probabilities >= best_threshold).astype(int).tolist()
        return {'results': [{'prediction': p, 'probability': prob} for p, prob in zip(predictions, probabilities)]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/explain/sample/{index}")
def explain_sample(index: int):
    return {"message": "SHAP explanation not available in API due to heavy computation. Please use Streamlit for SHAP."}
