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
    assert client.get("/api/v1/health").status_code == 200


def test_health_returns_model_info():
    body = client.get("/api/v1/health").json()
    assert 'model' in body
    assert 'threshold_f1' in body


def test_valid_prediction():
    r = client.post("/api/v1/predict", json=valid_data)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert 0.0 <= body["probability"] <= 1.0
    assert 0.0 <= body["threshold"] <= 1.0


def test_missing_feature():
    invalid = {k: v for k, v in valid_data.items() if k != 'age'}
    assert client.post("/api/v1/predict", json=invalid).status_code == 422


def test_invalid_job_rejected():
    inv = valid_data.copy()
    inv['job'] = 'not-a-real-job'
    assert client.post("/api/v1/predict", json=inv).status_code == 422


def test_invalid_marital_rejected():
    inv = valid_data.copy()
    inv['marital'] = 'banana'
    assert client.post("/api/v1/predict", json=inv).status_code == 422


def test_age_out_of_range():
    inv = valid_data.copy()
    inv['age'] = 150
    assert client.post("/api/v1/predict", json=inv).status_code == 422


def test_campaign_zero_rejected():
    inv = valid_data.copy()
    inv['campaign'] = 0
    assert client.post("/api/v1/predict", json=inv).status_code == 422


def test_batch_prediction():
    r = client.post("/api/v1/predict-batch", json={"data": [valid_data, valid_data]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_batch_size_limit():
    r = client.post("/api/v1/predict-batch", json={"data": [valid_data] * 5001})
    assert r.status_code == 413
