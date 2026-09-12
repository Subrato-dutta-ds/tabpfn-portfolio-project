from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

valid_data = {
    'age': 30, 'job': 'admin.', 'marital': 'single', 'education': 'university.degree',
    'default': 'no', 'housing': 'yes', 'loan': 'no', 'contact': 'cellular', 'month': 'may',
    'day_of_week': 'mon', 'campaign': 1, 'pdays': 999, 'previous': 0, 'poutcome': 'nonexistent',
    'emp_var_rate': 1.1, 'cons_price_idx': 93.994, 'cons_conf_idx': -36.4, 'euribor3m': 4.857,
    'nr_employed': 5191.0
}

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200

def test_valid_prediction():
    response = client.post("/api/v1/predict", json=valid_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert 0.0 <= response.json()["probability"] <= 1.0

def test_missing_feature():
    invalid_data = {k: v for k, v in valid_data.items() if k != 'age'}
    response = client.post("/api/v1/predict", json=invalid_data)
    assert response.status_code == 422

def test_batch_prediction():
    response = client.post("/api/v1/predict-batch", json={"data": [valid_data, valid_data]})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2

def test_batch_size_limit():
    # Test the max batch size (5000) is enforced
    large_batch = {"data": [valid_data] * 5001}
    response = client.post("/api/v1/predict-batch", json=large_batch)
    assert response.status_code == 413
