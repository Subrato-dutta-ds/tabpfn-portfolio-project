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
    return {"status": "ok", "model": metadata['model'], "threshold": float(best_threshold)}

@app.get("/health")
def health_check():
    return {"status": "ok", "model": metadata['model'], "threshold": float(best_threshold)}

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
        "cons.price.idx": data.cons_price_idx, "cons.conf.idx": data.cons_conf_idx,
        "euribor3m": data.euribor3m, "nr.employed": data.nr_employed
    }

@app.post('/predict')
def predict(data: CustomerFeatures):
    try:
        df = pd.DataFrame([map_to_dataset(data)])
        # CRITICAL FIX: Cast np.float32 to standard Python float!
        probability = float(pipeline.predict_proba(df)[0, 1])
        prediction = int(probability >= best_threshold)
        return {'prediction': prediction, 'probability': probability, 'threshold': float(best_threshold)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class BatchFeatures(BaseModel):
    data: List[CustomerFeatures]

@app.post('/predict-batch')
def predict_batch(batch: BatchFeatures):
    try:
        df = pd.DataFrame([map_to_dataset(item) for item in batch.data])
        # CRITICAL FIX: Cast the entire list to standard Python floats
        probabilities = [float(p) for p in pipeline.predict_proba(df)[:, 1]]
        predictions = [int(p >= best_threshold) for p in probabilities]
        return {'results': [{'prediction': p, 'probability': prob} for p, prob in zip(predictions, probabilities)]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
