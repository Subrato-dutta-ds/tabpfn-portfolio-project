from fastapi.testclient import TestClient
from src.api import app, map_to_dataset
import pytest

client = TestClient(app)

valid_data = {
    'age': 30, 'job': 'admin.', 'marital': 'single', 'education': 'university.degree',
    'default': 'no', 'housing': 'yes', 'loan': 'no', 'contact': 'cellular', 'month': 'may',
    'day_of_week': 'mon', 'campaign': 1, 'pdays': 999, 'previous': 0, 'poutcome': 'nonexistent',
    'emp_var_rate': 1.1, 'cons_price_idx': 93.994, 'cons_conf_idx': -36.4, 'euribor3m': 4.857,
    'nr_employed': 5191.0
}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

def test_valid_prediction():
    response = client.post("/predict", json=valid_data)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_missing_feature():
    invalid_data = {k: v for k, v in valid_data.items() if k != 'age'}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422 # Pydantic validation error

def test_batch_prediction():
    response = client.post("/predict-batch", json={"data": [valid_data, valid_data]})
    assert response.status_code == 200
    assert len(response.json()['results']) == 2
