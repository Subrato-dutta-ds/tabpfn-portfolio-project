"""
Utility functions for Streamlit app.
"""
import requests
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def check_api_health():
    """Check if FastAPI is running."""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def get_models():
    """Get list of available models from API."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get("models", []), data.get("default", "")
        return [], ""
    except:
        return [], ""

def get_features():
    """Get feature names from API."""
    try:
        response = requests.get(f"{API_URL}/features", timeout=2)
        if response.status_code == 200:
            return response.json().get("features", [])
        return []
    except:
        return []

def predict_single(features, model_name):
    """Call prediction API."""
    payload = {"features": features}
    try:
        start = time.time()
        response = requests.post(
            f"{API_URL}/predict?model={model_name}",
            json=payload,
            timeout=10
        )
        elapsed = time.time() - start
        if response.status_code == 200:
            return response.json(), elapsed
        return None, elapsed
    except:
        return None, 0

def predict_batch(features_list, model_name):
    """Call batch prediction API."""
    payload = {"features": features_list}
    try:
        response = requests.post(
            f"{API_URL}/predict/batch?model={model_name}",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_shap_explanation(index=0):
    """Get SHAP explanation for a sample."""
    try:
        response = requests.get(f"{API_URL}/explain/sample/{index}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def load_feature_names_fallback():
    """Load feature names from file if API fails."""
    try:
        with open("data/processed/feature_names.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    except:
        return []
