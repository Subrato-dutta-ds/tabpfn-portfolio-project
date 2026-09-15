from fastapi.testclient import TestClient
from src.api import app
import numpy as np
import pandas as pd
from src.business import rank_customers, expected_profit, optimal_k, campaign_summary

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

def test_valid_prediction():
    r = client.post("/api/v1/predict", json=valid_data)
    assert r.status_code == 200
    assert 0.0 <= r.json()["probability"] <= 1.0

def test_missing_feature():
    invalid = {k: v for k, v in valid_data.items() if k != 'age'}
    assert client.post("/api/v1/predict", json=invalid).status_code == 422

def test_invalid_job_rejected():
    inv = valid_data.copy(); inv['job'] = 'not-a-real-job'
    assert client.post("/api/v1/predict", json=inv).status_code == 422

def test_invalid_marital_rejected():
    inv = valid_data.copy(); inv['marital'] = 'banana'
    assert client.post("/api/v1/predict", json=inv).status_code == 422

def test_age_out_of_range():
    inv = valid_data.copy(); inv['age'] = 150
    assert client.post("/api/v1/predict", json=inv).status_code == 422

def test_campaign_zero_rejected():
    inv = valid_data.copy(); inv['campaign'] = 0
    assert client.post("/api/v1/predict", json=inv).status_code == 422

def test_batch_prediction():
    r = client.post("/api/v1/predict-batch", json={"data": [valid_data, valid_data]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2

def test_batch_size_limit():
    r = client.post("/api/v1/predict-batch", json={"data": [valid_data] * 5001})
    assert r.status_code == 413

def test_rank_customers_sorts_descending():
    df = pd.DataFrame({'customer_id': ['A', 'B', 'C'], 'probability': [0.3, 0.9, 0.5]})
    assert rank_customers(df)['customer_id'].tolist() == ['B', 'C', 'A']

def test_expected_profit_formula():
    assert expected_profit(0.5, 2000, 50) == 950.0
    assert expected_profit(0.0, 2000, 50) == -50.0

def test_optimal_k_positive_case():
    probs = np.array([0.9, 0.8, 0.01])
    k, profit = optimal_k(probs, revenue=2000, cost=50)
    assert k == 2 and profit > 0

def test_optimal_k_returns_zero_when_unprofitable():
    probs = np.array([0.001, 0.001, 0.001])
    k, profit = optimal_k(probs, revenue=2000, cost=50)
    assert k == 0
    assert profit == 0.0

def test_campaign_summary_k_zero():
    s = campaign_summary(np.array([0.9, 0.7, 0.5]), k=0, revenue=2000, cost=50)
    assert s['contacts'] == 0 and s['expected_profit'] == 0.0

def test_campaign_summary_metrics():
    probs = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    s = campaign_summary(probs, k=2, revenue=2000, cost=50)
    assert s['contacts'] == 2
    assert abs(s['expected_conversions'] - 1.6) < 1e-6
    assert s['lift'] > 1.0
