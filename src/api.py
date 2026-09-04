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
    # Fix #9: Return actual model info!
    return {"status": "ok", "model": metadata['model'], "threshold": best_threshold}

# ... (Rest of API logic omitted for brevity, but existing code works) ...
# Ensure your CustomerFeatures and /predict endpoints are there!
