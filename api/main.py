"""
FastAPI server for serving predictions.
Loads models at startup for low latency inference.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os
import pandas as pd
from typing import List
from api.schemas import (
    PredictionInput, PredictionOutput, 
    PredictionBatchInput, PredictionBatchOutput,
    HealthCheck, FeatureInfo
)
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="TabPFN Portfolio Project API",
    description="Model comparison API for Bank Marketing dataset",
    version="1.0.0"
)

# CORS middleware (allows frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODELS = {}
MODEL_NAMES = []
SCALER = None
FEATURE_NAMES = []
DEFAULT_MODEL = None
MODELS_DIR = "models"

@app.on_event("startup")
async def load_models():
    """Load all models and scaler at startup."""
    global MODELS, MODEL_NAMES, SCALER, FEATURE_NAMES, DEFAULT_MODEL
    
    print("=" * 60)
    print("🚀 Starting API Server - Loading Models...")
    print("=" * 60)
    
    try:
        # Load scaler
        scaler_path = f"{MODELS_DIR}/scaler.pkl"
        if os.path.exists(scaler_path):
            SCALER = joblib.load(scaler_path)
            print("✅ Loaded scaler")
        else:
            print("⚠️ Scaler not found")
        
        # Load feature names
        feature_path = f"{MODELS_DIR}/feature_names.pkl"
        if os.path.exists(feature_path):
            FEATURE_NAMES = joblib.load(feature_path)
            print(f"✅ Loaded {len(FEATURE_NAMES)} feature names")
        else:
            # Try text file
            txt_path = "data/processed/feature_names.txt"
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    FEATURE_NAMES = [line.strip() for line in f.readlines()]
                print(f"✅ Loaded {len(FEATURE_NAMES)} feature names from text file")
            else:
                print("⚠️ Feature names not found")
        
        # Load models
        model_files = {
            "Logistic Regression": "logistic_regression.pkl",
            "Random Forest": "random_forest.pkl",
            "XGBoost": "xgboost.pkl"
        }
        
        for name, filename in model_files.items():
            path = f"{MODELS_DIR}/{filename}"
            if os.path.exists(path):
                MODELS[name] = joblib.load(path)
                MODEL_NAMES.append(name)
                print(f"✅ Loaded: {name}")
            else:
                print(f"⚠️ {name} not found at {path}")
        
        if not MODELS:
            raise Exception("No models could be loaded!")
        
        # Set default model (prefer XGBoost)
        if "XGBoost" in MODELS:
            DEFAULT_MODEL = "XGBoost"
        else:
            DEFAULT_MODEL = MODEL_NAMES[0]
        
        print(f"\n📊 Default model: {DEFAULT_MODEL}")
        print(f"📊 Available models: {MODEL_NAMES}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise

@app.get("/", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy" if MODELS else "unhealthy",
        models_loaded=MODEL_NAMES,
        features_count=len(FEATURE_NAMES) if FEATURE_NAMES else 0
    )

@app.get("/models")
async def list_models():
    """List all available models."""
    return {"models": MODEL_NAMES, "default": DEFAULT_MODEL}

@app.get("/features", response_model=FeatureInfo)
async def list_features():
    """List feature names and their order."""
    return FeatureInfo(
        features=FEATURE_NAMES if FEATURE_NAMES else [],
        count=len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        description="Features in the exact order expected by the model"
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput, model: str = None):
    """
    Predict using a specified model.
    Use query param ?model=Random Forest to choose a model.
    """
    # Validate features length
    if FEATURE_NAMES and len(input_data.features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_NAMES)} features, got {len(input_data.features)}"
        )
    
    # Choose model
    if model and model in MODELS:
        model_name = model
    else:
        model_name = DEFAULT_MODEL
    
    model_obj = MODELS.get(model_name)
    if not model_obj:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available")
    
    # Convert and scale
    features = np.array(input_data.features).reshape(1, -1)
    if SCALER is not None:
        features = SCALER.transform(features)
    
    try:
        # Prediction
        pred = model_obj.predict(features)[0]
        
        # Probability
        if hasattr(model_obj, "predict_proba"):
            proba = model_obj.predict_proba(features)[0][1]
        else:
            proba = float(pred)
        
        # Confidence (percentage)
        confidence = proba * 100 if pred == 1 else (1 - proba) * 100
        
        return PredictionOutput(
            prediction=int(pred),
            probability=float(proba),
            model_used=model_name,
            confidence=float(confidence)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=PredictionBatchOutput)
async def predict_batch(input_data: PredictionBatchInput, model: str = None):
    """Batch predictions for multiple samples."""
    # Choose model
    if model and model in MODELS:
        model_name = model
    else:
        model_name = DEFAULT_MODEL
    
    model_obj = MODELS.get(model_name)
    if not model_obj:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available")
    
    # Convert and scale
    features = np.array(input_data.features)
    if SCALER is not None:
        features = SCALER.transform(features)
    
    try:
        predictions = model_obj.predict(features).tolist()
        
        if hasattr(model_obj, "predict_proba"):
            probabilities = model_obj.predict_proba(features)[:, 1].tolist()
        else:
            probabilities = predictions
        
        return PredictionBatchOutput(
            predictions=predictions,
            probabilities=probabilities,
            model_used=model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/explain/sample/{index}")
async def get_explanation(index: int = 0):
    """Get SHAP explanation for a specific sample."""
    shap_path = f"{MODELS_DIR}/shap_explainer.pkl"
    if not os.path.exists(shap_path):
        raise HTTPException(status_code=404, detail="SHAP explainer not found. Run src/explainability.py first.")
    
    try:
        X_test = np.load("data/processed/X_test.npy")
        explainer = joblib.load(shap_path)
        
        if index >= len(X_test):
            raise HTTPException(status_code=400, detail=f"Index {index} out of range (max {len(X_test)-1})")
        
        shap_values = explainer.shap_values(X_test[index].reshape(1, -1))
        
        return {
            "sample_index": index,
            "shap_values": shap_values[0].tolist(),
            "feature_names": FEATURE_NAMES if FEATURE_NAMES else [],
            "note": "Positive SHAP values push prediction toward class 1"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP error: {str(e)}")

@app.get("/reports")
async def list_reports():
    """List available reports and visualizations."""
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        return {"reports": []}
    
    files = [f for f in os.listdir(reports_dir) if f.endswith(('.png', '.csv', '.txt'))]
    return {
        "reports": files,
        "path": f"/static/{reports_dir}/"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
