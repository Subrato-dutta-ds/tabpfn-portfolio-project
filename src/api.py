import os, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))

app = FastAPI()

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

@app.post('/predict')
def predict(data: CustomerFeatures):
    try:
        df = pd.DataFrame([data.model_dump()])
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0].tolist()
        return {'prediction': int(prediction), 'probability': probability}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
from pydantic import BaseModel
from typing import List

class BatchFeatures(BaseModel):
    data: List[CustomerFeatures]

@app.post('/predict-batch')
def predict_batch(batch: BatchFeatures):
    try:
        df = pd.DataFrame([item.model_dump() for item in batch.data])
        predictions = pipeline.predict(df).tolist()
        probabilities = pipeline.predict_proba(df).tolist()
        return {'results': [{'prediction': int(p), 'probability': prob} for p, prob in zip(predictions, probabilities)]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
