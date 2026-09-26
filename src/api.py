import os, joblib, json, logging, sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'production_model_active.pkl')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'production_model.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata_active.json')
if not os.path.exists(METADATA_PATH):
    METADATA_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata.json')

LOG_DIR = Path(BASE_DIR) / 'logs'
LOG_DIR.mkdir(exist_ok=True)
PREDICTION_LOG = LOG_DIR / 'predictions.jsonl'


def _log_prediction(features, probability, prediction):
    try:
        entry = {'timestamp': datetime.utcnow().isoformat() + 'Z',
                 'features': features, 'probability': probability, 'prediction': prediction}
        with open(PREDICTION_LOG, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        logger.warning("Failed to log prediction")


def _load_artifacts():
    missing = [p for p in (MODEL_PATH, METADATA_PATH) if not os.path.exists(p)]
    if missing:
        logger.error("Missing artifacts: " + str(missing))
        logger.error("Run: python src/generate_campaign_data.py && python src/train_models.py")
        sys.exit(1)
    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    return pipeline, meta


pipeline, metadata = _load_artifacts()
best_threshold = metadata['threshold_f1']

from src.schema import API_ALIASES, FEATURE_SCHEMA

app = FastAPI(title='Bank Marketing Campaign API', version='1.0.0')
NUMERIC = FEATURE_SCHEMA['numerical']
CATEGORICAL = FEATURE_SCHEMA['categorical']


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
    def validate_cat(cls, v, info):
        allowed = CATEGORICAL.get(info.field_name)
        if allowed and v not in allowed:
            raise ValueError(f'{info.field_name} must be one of {allowed}')
        return v

    @field_validator('age', 'campaign', 'pdays', 'previous', 'emp_var_rate',
                     'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed')
    @classmethod
    def validate_num(cls, v, info):
        spec = NUMERIC[API_ALIASES.get(info.field_name, info.field_name)]
        if not (spec['min'] <= v <= spec['max']):
            raise ValueError(f'{info.field_name} out of range')
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


def map_to_dataset(data):
    raw = data.model_dump()
    return {API_ALIASES.get(k, k): v for k, v in raw.items()}


@app.get('/api/v1/health')
def health():
    return {'status': 'ok', 'model': metadata['model'],
            'threshold_f1': float(best_threshold),
            'calibration': metadata.get('calibration', 'none'),
            'git_sha': metadata.get('git_sha', 'unknown')}


@app.post('/api/v1/predict', response_model=PredictionResponse)
def predict(data: CustomerFeatures):
    try:
        df = pd.DataFrame([map_to_dataset(data)])
        probability = float(pipeline.predict_proba(df)[0, 1])
        prediction = int(probability >= best_threshold)
        _log_prediction(map_to_dataset(data), probability, prediction)
        return PredictionResponse(prediction=prediction, probability=probability,
                                  threshold=float(best_threshold))
    except Exception:
        logger.exception('Prediction failed')
        raise HTTPException(status_code=500, detail='Prediction service failed.')


@app.post('/api/v1/predict-batch', response_model=BatchPredictionResponse)
def predict_batch(batch: BatchFeatures):
    if len(batch.data) > 5000:
        raise HTTPException(status_code=413, detail='Maximum batch size is 5000')
    try:
        records = [map_to_dataset(i) for i in batch.data]
        df = pd.DataFrame(records)
        probs = [float(p) for p in pipeline.predict_proba(df)[:, 1]]
        preds = [int(p >= best_threshold) for p in probs]
        return BatchPredictionResponse(
            results=[PredictionResponse(prediction=p, probability=pr, threshold=float(best_threshold))
                     for p, pr in zip(preds, probs)],
            total=len(preds))
    except Exception:
        logger.exception('Batch failed')
        raise HTTPException(status_code=500, detail='Batch prediction service failed.')
