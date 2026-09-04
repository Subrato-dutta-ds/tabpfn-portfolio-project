import os, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))

app = FastAPI()

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
    return {
        "age": data.age, "job": data.job, "marital": data.marital, "education": data.education,
        "default": data.default, "housing": data.housing, "loan": data.loan, "contact": data.contact,
        "month": data.month, "day_of_week": data.day_of_week, "campaign": data.campaign, "pdays": data.pdays,
        "previous": data.previous, "poutcome": data.poutcome, "emp.var.rate": data.emp_var_rate,
        "cons.price.idx": data.cons_price_idx, "cons.conf.idx": data.cons_conf_idx, "euribor3m": data.euribor3m,
        "nr.employed": data.nr_employed
    }

@app.post('/predict')
def predict(data: CustomerFeatures):
    try:
        df = pd.DataFrame([map_to_dataset(data)])
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0].tolist()
        return {'prediction': int(prediction), 'probability': probability}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class BatchFeatures(BaseModel):
    data: List[CustomerFeatures]

@app.post('/predict-batch')
def predict_batch(batch: BatchFeatures):
    try:
        df = pd.DataFrame([map_to_dataset(item) for item in batch.data])
        predictions = pipeline.predict(df).tolist()
        probabilities = pipeline.predict_proba(df).tolist()
        return {'results': [{'prediction': int(p), 'probability': prob} for p, prob in zip(predictions, probabilities)]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
