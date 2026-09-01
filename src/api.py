import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI()

try:
    pipeline = joblib.load("models/model_pipeline.pkl")
except FileNotFoundError:
    raise RuntimeError("Pipeline not found! Run 'python src/train_models.py' first.")

@app.post("/predict")
async def predict(features: dict):
    try:
        df = pd.DataFrame([features])
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0].tolist()
        return {"prediction": int(prediction), "probability": probability}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))