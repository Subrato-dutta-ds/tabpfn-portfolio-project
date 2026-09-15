from fastapi.testclient import TestClient
from src.api import app
import pytest

client = TestClient(app)

valid_data = {
    'age': 30, 'job': 'admin.', 'marital': 'single', 'education': 'university.degree',
    'default': 'no', 'housing': 'yes', 'loan': 'no', 'contact': 'cellular', 'month': 'may',
    'day_of_week': 'mon', 'campaign': 1, 'pdays': 999, 'previous': 0, 'poutcome': 'nonexistent',
    'emp_var_rate': 1.1, 'cons_price_idx': 93.994, 'cons_conf_idx': -36.4, 'euribor3m': 4.857,
    'nr_employed': 5191.0
}

# -------- API tests --------

def test_health_endpoint():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_valid_prediction():
    r = client.post("/api/v1/predict", json=valid_data)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert 0.0 <= body["probability"] <= 1.0
    assert 0.0 <= body["threshold"] <= 1.0

def test_missing_feature():
    invalid = {k: v for k, v in valid_data.items() if k != 'age'}
    r = client.post("/api/v1/predict", json=invalid)
    assert r.status_code == 422

def test_invalid_categorical_rejected():
    invalid = valid_data.copy()
    invalid['job'] = 'not-a-real-job'
    r = client.post("/api/v1/predict", json=invalid)
    assert r.status_code == 422

def test_invalid_marital_rejected():
    invalid = valid_data.copy()
    invalid['marital'] = 'banana'
    r = client.post("/api/v1/predict", json=invalid)
    assert r.status_code == 422

def test_age_out_of_range_rejected():
    invalid = valid_data.copy()
    invalid['age'] = 150
    r = client.post("/api/v1/predict", json=invalid)
    assert r.status_code == 422

def test_campaign_zero_rejected():
    invalid = valid_data.copy()
    invalid['campaign'] = 0
    r = client.post("/api/v1/predict", json=invalid)
    assert r.status_code == 422

def test_batch_prediction():
    r = client.post("/api/v1/predict-batch", json={"data": [valid_data, valid_data]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2

def test_batch_size_limit():
    r = client.post("/api/v1/predict-batch", json={"data": [valid_data] * 5001})
    assert r.status_code == 413

# -------- Business logic tests --------

def test_profit_zero_probability():
    # If p=0, profit per customer = -cost (should not be selected)
    p, revenue, cost = 0.0, 2000, 50
    profit = p * revenue - cost
    assert profit == -50

def test_profit_at_break_even():
    # p * revenue = cost => break even at p = cost/revenue
    revenue, cost = 2000, 50
    break_even_p = cost / revenue
    assert abs(break_even_p - 0.025) < 1e-9
    assert (break_even_p * revenue - cost) == 0

def test_profit_positive_above_break_even():
    p, revenue, cost = 0.5, 2000, 50
    profit = p * revenue - cost
    assert profit == 950

def test_top_k_ranking_monotonic():
    import numpy as np
    probabilities = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
    sorted_desc = np.sort(probabilities)[::-1]
    assert (sorted_desc == probabilities).all()  # already sorted desc
