import os, joblib, json, logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'production_model.pkl'))
with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json')) as f:
    metadata = json.load(f)
best_threshold = metadata['threshold_f1']

from src.schema import API_ALIASES, FEATURE_SCHEMA

app = FastAPI(title='Bank Marketing Campaign API', version='1.0.0')

NUMERIC = FEATURE_SCHEMA['numerical']
CATEGORICAL = FEATURE_SCHEMA['categorical']

def _numeric_field(col_api):
    ds_name = API_ALIASES.get(col_api, col_api)
    return NUMERIC[ds_name]

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

    @field_validator('job', 'marital', 'education', 'default', 'housing', 'loan',
                     'contact', 'month', 'day_of_week', 'poutcome')
    @classmethod
    def validate_categorical(cls, v, info):
        field = info.field_name
        allowed = CATEGORICAL.get(field)
        if allowed and v not in allowed:
            raise ValueError(f'{field} must be one of {allowed}')
        return v

    @field_validator('age', 'campaign', 'pdays', 'previous', 'emp_var_rate',
                     'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed')
    @classmethod
    def validate_numeric(cls, v, info):
        field = info.field_name
        spec = _numeric_field(field)
        if not (spec['min'] <= v <= spec['max']):
            raise ValueError(f'{field} must be between {spec["min"]} and {spec["max"]}')
        return v

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float

class BatchFeatures(BaseModel):
    data: List[CustomerFeatures]

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int

def map_to_dataset(data: CustomerFeatures):
    raw = data.model_dump()
    return {API_ALIASES.get(k, k): v for k, v in raw.items()}

@app.get('/api/v1/health')
def health_check():
    return {
        'status': 'ok',
        'model': metadata['model'],
        'threshold_f1': float(best_threshold),
        'calibration': metadata.get('calibration', 'none'),
        'model_version': metadata.get('model_version', 'unknown'),
    }

@app.post('/api/v1/predict', response_model=PredictionResponse)
def predict(data: CustomerFeatures):
    try:
        df = pd.DataFrame([map_to_dataset(data)])
        probability = float(pipeline.predict_proba(df)[0, 1])
        prediction = int(probability >= best_threshold)
        return PredictionResponse(prediction=prediction, probability=probability, threshold=float(best_threshold))
    except Exception:
        logger.exception('Prediction failed')
        raise HTTPException(status_code=500, detail='Prediction service failed.')

@app.post('/api/v1/predict-batch', response_model=BatchPredictionResponse)
def predict_batch(batch: BatchFeatures):
    if len(batch.data) > 5000:
        raise HTTPException(status_code=413, detail='Maximum batch size is 5000')
    try:
        df = pd.DataFrame([map_to_dataset(item) for item in batch.data])
        probabilities = [float(p) for p in pipeline.predict_proba(df)[:, 1]]
        predictions = [int(p >= best_threshold) for p in probabilities]
        results = [PredictionResponse(prediction=p, probability=prob, threshold=float(best_threshold))
                   for p, prob in zip(predictions, probabilities)]
        return BatchPredictionResponse(results=results, total=len(results))
    except Exception:
        logger.exception('Batch prediction failed')
        raise HTTPException(status_code=500, detail='Batch prediction service failed.')
